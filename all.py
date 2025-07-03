main.py
import torch
import yaml
import optuna
import os
import pandas as pd
import numpy as np
import inspect

from src.data_loader.data_processor import load_and_preprocess_data, TOCDataset, DataLoader
from src.model.model import (
    LSTM_Model, 
    TCN_Model, 
    TCN_LSTM_Attention_Model, 
    WeightedMSELoss
)
from src.train.trainer import run_optuna_objective, train_model, evaluate_model
from src.visualization.plot import (
    plot_loss_curves, 
    plot_actual_vs_predicted, 
)


def main():
    # ===== 0. 全局设置 & 加载配置 =====
    try:
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("错误: config.yaml 文件未找到。请确保它在 'src/config/' 目录下。")
        return

    # 提取当前运行的配置
    run_config = config['run_config']
    model_name = run_config['model_name']
    
    print(f"--- 使用配置文件: src/config/config.yaml ---")
    print(f"--- 欲运行的模型: {model_name} ---")
    print(f"--- 使用设备: {run_config['device']} ---")


    # ===== 1. 数据加载与预处理 =====
    print("\n[步骤 1/5] 加载并预处理数据...")
    train_dataset, val_dataset, test_dataset, output_feature_scaler, input_size, updated_config, train_scaled_original, test_scaled_original = load_and_preprocess_data(run_config)
    run_config.update(updated_config) 



    # ===== 2. 超参数调优 (Optuna) =====
    print(f"\n[步骤 2/5] 开始为 '{model_name}' 模型进行超参数调优...")
    # run_optuna_objective 内部调用了重构后的 train_model 函数
    objective_wrapper = lambda trial: run_optuna_objective(
        trial, 
        train_dataset, 
        val_dataset, 
        model_name,
        input_size, 
        run_config
    )
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective_wrapper, n_trials=run_config.get('n_trials', 50), timeout=run_config.get('timeout', 7200))

    print("\n超参数调优完成!")
    best_params = study.best_trial.params
    print(f"最佳试验的验证损失: {study.best_trial.value:.6f}")
    print(f"最佳试验的超参数: {best_params}")

    # 保存调优结果
    results_dir = run_config.get('results_dir', 'results')
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    study.trials_dataframe().to_csv(os.path.join(results_dir, f'tuning_results_{model_name}.csv'))


    # ===== 3. 训练最终模型 =====
    print(f"\n[步骤 3/5] 使用最佳超参数训练最终的 '{model_name}' 模型...")

    # 根据名称动态创建模型
    MODELS = {
        'lstm': LSTM_Model,
        'tcn': TCN_Model,
        'tcn_lstm_attention': TCN_LSTM_Attention_Model
    }
    
    if model_name not in MODELS:
        raise ValueError(f"未知的模型名称: '{model_name}'。请在 main.py 的 MODELS 字典中注册它。")
    
    ModelClass = MODELS[model_name]

    # 从 best_params 中筛选出当前模型构造函数真正需要的参数
    constructor_args = inspect.signature(ModelClass.__init__).parameters.keys()
    final_model_params = {k: v for k, v in best_params.items() if k in constructor_args}
    final_model_params['input_size'] = input_size # 确保 input_size 被传入

    if model_name == 'tcn':
        final_model_params['output_size'] = 1 

    # 实例化最终模型
    final_model = ModelClass(**final_model_params).to(run_config['device'])
    print("\n最终模型结构:")
    print(final_model)

    # 准备数据加载器
    train_loader_final = DataLoader(train_dataset, batch_size=best_params.get('batch_size', 64), shuffle=True)
    val_loader_final = DataLoader(val_dataset, batch_size=best_params.get('batch_size', 64), shuffle=False)
    test_loader_final = DataLoader(test_dataset, batch_size=best_params.get('batch_size', 64), shuffle=False)
    
    # 定义损失函数和优化器
    criterion = WeightedMSELoss(
        warning_threshold=run_config['scaled_warning_threshold'],
        weight_factor=best_params.get('loss_weight_factor', 2.0)
    )
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_params.get('learning_rate', 1e-3),
        weight_decay=best_params.get('weight_decay', 1e-4)
    )

    # 训练模型
    _, train_losses_final, val_losses_final = train_model(
        final_model,
        train_loader_final,
        val_loader_final,
        criterion,
        final_optimizer,
        run_config,
        epoch_offset=0, # 从epoch 0开始
        trial=None      # 不是Optuna trial
    )
    print("\n最终模型训练完成!")


    # ===== 4. 评估模型 =====
    print("\n[步骤 4/5] 评估最终模型...")
    # 确保模型保存目录存在
    model_save_dir = os.path.dirname(run_config['model_save_path'])
    if model_save_dir and not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    final_model.load_state_dict(torch.load(run_config['model_save_path'])) # 加载最佳权重
    
    # 动态构建测试集结果的CSV输出路径
    results_dir = run_config['results_dir']
    csv_base_name = run_config.get('results_csv_name', 'predictions.csv')
    test_csv_filename = f"{model_name}_test_{csv_base_name}"
    test_csv_path = os.path.join(results_dir, test_csv_filename)
    print(f"测试集评估结果将保存到: {test_csv_path}")
    
    all_predictions_test_denorm, all_true_values_test_denorm, _, _, _ = evaluate_model(
        final_model, 
        test_loader_final, 
        output_feature_scaler, 
        run_config,
        output_csv_path=test_csv_path 
    )
    # calculate_warning_metrics(...) # 您可以按需调用


    # ===== 5. 可视化 =====
    print("\n[步骤 5/5] 生成可视化结果...")
    plot_loss_curves(train_losses_final, val_losses_final, results_dir, title_suffix=f"({model_name} Final Model)")

    # 为了在完整数据集上绘图，重新准备 full_loader
    full_scaled_data = np.vstack((train_scaled_original, test_scaled_original))
    from src.data_loader.data_processor import TOCDataset as FullTOCDataset # 避免混淆
    
    if not pd.api.types.is_datetime64_any_dtype(full_timestamps):
        full_timestamps = pd.to_datetime(full_timestamps)
        
    full_dataset = TOCDataset(full_scaled_data,
                            full_timestamps, # <-- 使用真实的日期时间戳
                            val_dataset.input_features_idx,  # 从验证集或测试集获取索引
                            val_dataset.output_feature_idx,
                            run_config['sequence_length'],
                            run_config['forecast_horizon'])
    full_loader = DataLoader(full_dataset, batch_size=best_params.get('batch_size', 64), shuffle=False)

    # 为全量数据预测构建独立的CSV路径
    full_csv_filename = f"{model_name}_full_data_{csv_base_name}"
    full_csv_path = os.path.join(results_dir, full_csv_filename)
    print(f"全量数据评估结果将保存到: {full_csv_path}")

    # 再次调用评估函数，并传入新的CSV路径
    all_predictions_full_denorm, all_true_values_full_denorm, _, _, _ = evaluate_model(
        final_model, 
        full_loader, 
        output_feature_scaler, 
        run_config,
        output_csv_path=full_csv_path # 传入为全量数据构建的路径
    )

    original_train_samples_count = len(train_scaled_original) - run_config['sequence_length'] - run_config['forecast_horizon'] + 1
    train_val_split_idx = original_train_samples_count
    train_split_idx = original_train_samples_count

    # 分割点绘图
    plot_actual_vs_predicted(all_true_values_full_denorm, all_predictions_full_denorm, 
                            run_config['warning_threshold'], train_split_idx, results_dir, 
                            title_suffix=f"({model_name} All Data)")
    
    print(f"\n程序运行结束。所有结果已保存到 '{results_dir}' 目录。")


if __name__ == '__main__':
    main()

data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

# ===================================================================
#  Dataset 定义
# ===================================================================

class TOCDataset(Dataset):
    """
    一个标准的PyTorch数据集，用于处理时间序列数据。
    它会根据给定的索引，懒加载（on-the-fly）地创建每一个输入序列和目标值。

    ## [MODIFIED] ##
    - 增加了 `timestamps` 属性，用于存储与每个数据点对应的时间戳。
    - 这对于后续评估和可视化至关重要。
    """
    def __init__(self, data, timestamps, input_features_idx, output_feature_idx, sequence_length, forecast_horizon):
        self.data = data
        # 时间戳应与 'data' 的行一一对应
        self.timestamps = timestamps 
        self.input_features_idx = input_features_idx
        self.output_feature_idx = output_feature_idx
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        # 可生成的样本数量
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # 输入序列 (Input sequence)
        input_seq = self.data[idx : idx + self.sequence_length, self.input_features_idx].T
        
        # 目标值 (Target value)
        target_toc = self.data[idx + self.sequence_length + self.forecast_horizon - 1, self.output_feature_idx]

        # 返回PyTorch张量
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_toc, dtype=torch.float32)

class PrecomputedTOCDataset(Dataset):
    """
    一个预计算的数据集，用于存储已经处理好（例如，通过数据增强）的样本。
    这避免了在每个epoch都重新计算，但会占用更多内存。

    ## [MODIFIED] ##
    - 同样增加了 `timestamps` 属性，以保持与TOCDataset的兼容性。
    """
    def __init__(self, inputs, targets, timestamps):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.timestamps = timestamps # 存储与样本匹配的时间戳

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ===================================================================
#  数据加载与预处理主函数
# ===================================================================

def load_and_preprocess_data(config):
    """
    加载、预处理、增强并准备数据集的主函数。
    
    ## [HEAVILY MODIFIED] ##
    - 替换SMOTE: 现在使用专门为时间序列设计的“Jittering”（添加噪声）方法来增强少数类（高TOC值）样本。
      这种方法在不破坏时间依赖性的前提下创建新的、逼真的样本。
    - 修复时间戳问题: 整个流程现在都会处理 `timestamp` 列，并将其正确地传递给Dataset对象。
    """
    # --- 1. 加载和排序 ---
    df = pd.read_csv(config['data_path'])
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        print("数据已根据 'timestamp' 列进行时序排序。")
    else:
        print("警告: 未找到 'timestamp' 列，请确保数据已按时间顺序排列。")

    # 分离出特征和时间戳
    timestamps = df['timestamp']
    features_df = df[config['input_features'] + [config['output_feature']]]

    # --- 2. 三向数据分割 ---
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    
    train_size = int(len(features_df) * train_ratio)
    val_size = int(len(features_df) * val_ratio)
    
    train_df = features_df.iloc[:train_size]
    val_df = features_df.iloc[train_size : train_size + val_size]
    test_df = features_df.iloc[train_size + val_size:]

    # 相应地分割时间戳
    train_timestamps_df = timestamps.iloc[:train_size]
    val_timestamps_df = timestamps.iloc[train_size : train_size + val_size]
    test_timestamps_df = timestamps.iloc[train_size + val_size:]
    
    print(f"Data split: Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)})")

    # --- 3. 数据缩放 ---
    scaler = StandardScaler() if config['scaler_type'] == 'Standard' else MinMaxScaler()
    scaler.fit(train_df) 
    
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df) 
    test_scaled = scaler.transform(test_df) 

    # 为输出特征创建一个独立的缩放器，用于后续的反归一化
    output_feature_scaler = StandardScaler() if config['scaler_type'] == 'Standard' else MinMaxScaler()
    output_feature_scaler.fit(train_df[[config['output_feature']]])

    # 计算缩放后的预警阈值
    config['scaled_warning_threshold'] = output_feature_scaler.transform(
        np.array([[config['warning_threshold']]])
    ).flatten()[0]
    print(f"归一化后的预警阈值: {config['scaled_warning_threshold']:.4f}")

    # 获取特征列的索引
    input_feature_indices = [features_df.columns.get_loc(col) for col in config['input_features']]
    output_feature_index = features_df.columns.get_loc(config['output_feature'])

    # --- 4. 创建初始序列 & 准备数据增强 ---
    # 将完整的训练数据转换成序列样本
    all_train_inputs, all_train_targets, all_train_timestamps = [], [], []
    
    # 为每个样本找到其对应的时间戳（目标值的时间戳）
    seq_len, horizon = config['sequence_length'], config['forecast_horizon']
    
    for i in range(len(train_scaled) - seq_len - horizon + 1):
        input_data = train_scaled[i : i + seq_len, input_feature_indices].T
        target_data = train_scaled[i + seq_len + horizon - 1, output_feature_index]
        timestamp_data = train_timestamps_df.iloc[i + seq_len + horizon - 1]

        all_train_inputs.append(input_data)
        all_train_targets.append(target_data)
        all_train_timestamps.append(timestamp_data)

    all_train_inputs = np.array(all_train_inputs)
    all_train_targets = np.array(all_train_targets)
    all_train_timestamps = np.array(all_train_timestamps)
    
    # --- 5. 时间序列数据增强 ---
    # 识别出少数类（高TOC值）
    is_minority = all_train_targets >= config['scaled_warning_threshold']
    minority_inputs = all_train_inputs[is_minority]
    minority_targets = all_train_targets[is_minority]
    minority_timestamps = all_train_timestamps[is_minority]
    
    num_minority = len(minority_inputs)
    num_majority = len(all_train_inputs) - num_minority
    print(f"数据增强前: 训练集样本总数: {len(all_train_inputs)}")
    print(f"其中，超标样本数 (少数类): {num_minority}, 正常样本数 (多数类): {num_majority}")

    # 仅当少数类样本存在且数量少于多数类时，才进行数据增强
    if 1 < num_minority < num_majority:
        # 计算需要生成多少新样本以达到平衡
        augmentation_factor = (num_majority // num_minority) - 1
        noise_level = config.get('augmentation_noise_level', 0.02) # 从配置中读取噪声等级，或使用默认值

        if augmentation_factor > 0:
            print(f"执行时间序列数据增强 (Jittering)，每个少数类样本将生成 {augmentation_factor} 个新样本。")
            augmented_inputs_list, augmented_targets_list, augmented_timestamps_list = [], [], []
            
            for i in range(augmentation_factor):
                # 为每个少数类样本添加高斯噪声 (Jittering)
                noise = np.random.normal(loc=0.0, scale=noise_level, size=minority_inputs.shape)
                augmented_inputs_list.append(minority_inputs + noise)
                augmented_targets_list.append(minority_targets) # 目标值保持不变
                augmented_timestamps_list.append(minority_timestamps) # 时间戳也保持不变

            # 将生成的新样本与原始样本合并
            augmented_inputs = np.concatenate(augmented_inputs_list, axis=0)
            augmented_targets = np.concatenate(augmented_targets_list, axis=0)
            augmented_timestamps = np.concatenate(augmented_timestamps_list, axis=0)

            final_train_inputs = np.vstack([all_train_inputs, augmented_inputs])
            final_train_targets = np.hstack([all_train_targets, augmented_targets])
            final_train_timestamps = np.hstack([all_train_timestamps, augmented_timestamps])

            # 打乱合并后的数据集，确保模型训练的随机性
            shuffled_indices = np.random.permutation(len(final_train_inputs))
            final_train_inputs = final_train_inputs[shuffled_indices]
            final_train_targets = final_train_targets[shuffled_indices]
            final_train_timestamps = final_train_timestamps[shuffled_indices]

            print(f"数据增强后: 训练集样本总数: {len(final_train_targets)}")
        else:
            print("多数类与少数类样本数量差距不大，不执行数据增强。")
            final_train_inputs, final_train_targets, final_train_timestamps = all_train_inputs, all_train_targets, all_train_timestamps
    else:
        print("少数类样本过少或过多，不执行数据增强。")
        final_train_inputs, final_train_targets, final_train_timestamps = all_train_inputs, all_train_targets, all_train_timestamps

    # --- 6. 创建最终的 Dataset 对象 ---
    train_dataset = PrecomputedTOCDataset(final_train_inputs, final_train_targets, final_train_timestamps)
    
    val_dataset = TOCDataset(
        val_scaled, 
        val_timestamps_df, 
        input_feature_indices, 
        output_feature_index, 
        config['sequence_length'], 
        config['forecast_horizon']
    )
    
    test_dataset = TOCDataset(
        test_scaled, 
        test_timestamps_df,
        input_feature_indices, 
        output_feature_index, 
        config['sequence_length'], 
        config['forecast_horizon']
    )

    print(f"最终训练集大小 (增强后): {len(train_dataset)}")
    print(f"最终验证集大小: {len(val_dataset)}")
    print(f"最终测试集大小: {len(test_dataset)}")

    return timestamps, train_dataset, val_dataset, test_dataset, output_feature_scaler, len(config['input_features']), config, train_scaled, test_scaled

config.yaml
# ===================================================================
#  运行配置
# ===================================================================
run_config:
  # 运行模型： 'lstm', 'tcn', 'tcn_lstm_attention'
  model_name: 'lstm' 
  
  # --- 数据与预处理 ---
  data_path: 'data/processed/TOC_dataset.csv'   # 已时移
  output_feature: 'UPW'
  input_features: ['WW', 'RW', 'DI', 'RO', 'MD']
  sequence_length: 24
  forecast_horizon: 1
  train_ratio: 0.7  # 70% 训练
  val_ratio: 0.15   # 15% 验证, 15% 测试
  scaler_type: 'Standard' #  'Standard', 'MinMax'
  augmentation_noise_level: 0.02 # 数据增强时添加的噪声大小
  
  # --- 训练参数 ---
  device: 'cuda' #  'cpu'
  num_epochs: 100
  early_stopping_patience: 20
  
  # --- 损失函数与预警 ---
  warning_threshold: 0.65
  
  # --- Optuna 调优参数 ---
  n_trials: 50      # Optuna 试验次数
  timeout: 7200     # Optuna 运行时间上限 (秒)
  optuna_epochs: 30 # Optuna 调优时，每个 trial 训练的 epoch 数
  
  # --- 路径设置 ---
  model_save_path: 'outputs/checkpoints/best_model_LSTM.pth'
  results_csv_name: 'predictions.csv'
  results_dir: 'outputs/results'
  
# ===================================================================
#  模型参数库
# ===================================================================
model_params:
  # -------------------------
  # 模型1: lstm
  # -------------------------
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    output_size: 1

  # -------------------------
  # 模型2: tcn
  # -------------------------
  tcn:
    num_channels: [32, 64, 128]
    kernel_size: 3
    dropout: 0.2

  # -------------------------
  # 模型3: tcn_lstm_attention
  # -------------------------
  tcn_lstm_attention:
    tcn_num_channels: [30, 60]
    tcn_kernel_size: 3
    tcn_dropout: 0.25
    lstm_hidden_size: 128
    lstm_num_layers: 2
    lstm_dropout: 0.3

model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- TCN 基础模块 ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# --- Attention 模块 ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, rnn_outputs, final_hidden_state):
        final_hidden_state_expanded = final_hidden_state.unsqueeze(1).repeat(1, rnn_outputs.size(1), 1)
        combined = torch.cat((rnn_outputs, final_hidden_state_expanded), dim=2)
        attn_weights = self.v(torch.tanh(self.attn(combined)))
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * rnn_outputs, dim=1)
        return context_vector, attn_weights

# ==========================================================
# --- 模型库 (Model Zoo) ---
# ==========================================================

# 模型1: 单一LSTM模型
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(LSTM_Model, self).__init__()
        # LSTM需要 [Batch, SeqLen, Features]
        self.input_format = 'NLC' 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out.squeeze(1)

# 模型2: 单一TCN模型
class TCN_Model(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_Model, self).__init__()
        # TCN需要 [Batch, Features, SeqLen]
        self.input_format = 'NCL' 
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        tcn_out = self.tcn(x)
        # 使用最后一个时间步的输出进行预测
        last_time_step_out = tcn_out[:, :, -1]
        out = self.fc(last_time_step_out)
        return out.squeeze(1)

# 模型3: TCN-LSTM-Attention 复合模型
class TCN_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, tcn_num_channels, tcn_kernel_size, tcn_dropout,
                 lstm_hidden_size, lstm_num_layers, lstm_dropout):
        super(TCN_LSTM_Attention_Model, self).__init__()
        # 复合模型输入先给TCN，所以是 'NCL'
        self.input_format = 'NCL'
        self.tcn = TemporalConvNet(input_size, tcn_num_channels, tcn_kernel_size, tcn_dropout)
        tcn_output_channels = tcn_num_channels[-1]
        self.lstm = nn.LSTM(tcn_output_channels, lstm_hidden_size, lstm_num_layers,
                              batch_first=True, dropout=lstm_dropout if lstm_num_layers > 1 else 0)
        self.attention = Attention(lstm_hidden_size)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        tcn_output = self.tcn(x)
        lstm_input = tcn_output.permute(0, 2, 1)
        rnn_outputs, (hn, cn) = self.lstm(lstm_input)
        final_hidden_state = hn[-1, :, :]
        context_vector, _ = self.attention(rnn_outputs, final_hidden_state)
        out = self.fc(context_vector)
        return out.squeeze(1)

# --- 自定义损失函数 ---
class WeightedMSELoss(nn.Module):
    def __init__(self, warning_threshold: float, weight_factor: float):
        super(WeightedMSELoss, self).__init__()
        self.warning_threshold = warning_threshold
        self.weight_factor = weight_factor
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        weights = torch.ones_like(targets)
        high_toc_indices = targets >= self.warning_threshold
        weights[high_toc_indices] = self.weight_factor
        weighted_loss = mse_loss * weights
        return weighted_loss.mean()
    
trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import os
import pandas as pd

from src.model.model import (
    LSTM_Model, 
    TCN_Model, 
    TCN_LSTM_Attention_Model, 
    WeightedMSELoss
)
from src.data_loader.data_processor import TOCDataset # 导入TOCDataset用于类型检查
from src.utils.helpers import adjust_input_for_model

# 动态实例化模型
MODELS = {
    'lstm': LSTM_Model,
    'tcn': TCN_Model,
    'tcn_lstm_attention': TCN_LSTM_Attention_Model
}

def train_model(model, train_loader, val_loader, criterion, optimizer, config, epoch_offset=0, trial=None):
    """
    一个通用的模型训练函数，经过重构以同时支持最终训练和Optuna调优。

    - `trial` (optuna.Trial, optional): 
      - 如果提供了 `trial` 对象，函数会进入 "Optuna模式":
        1. 训练轮数由 `config['optuna_epochs']` 控制。
        2. 在每轮结束后，向Optuna报告验证损失 (`trial.report`)。
        3. 检查是否应该剪枝 (`trial.should_prune`) 以提前终止不佳的试验。
        4. **不会**执行早停逻辑或保存模型文件。
      - 如果 `trial` 为 `None`，函数进入 "最终训练模式":
        1. 训练轮数由 `config['num_epochs']` 控制。
        2. 执行早停逻辑 (`early_stopping_patience`)。
        3. 当验证损失改善时，保存最佳模型。
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    # 根据是否在Optuna模式下运行，决定训练的总轮数
    num_epochs = config.get('optuna_epochs', 30) if trial else config['num_epochs']
    
    # 确定显示时总轮数的字符串
    total_epochs_str = config.get('optuna_epochs') if trial else config['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            inputs = adjust_input_for_model(inputs, model)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # --- 验证阶段 ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                inputs = adjust_input_for_model(inputs, model)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # 仅在非Optuna模式下打印日志，避免刷屏
        if not trial:
            print(f"Epoch [{epoch_offset + epoch + 1}/{total_epochs_str}], Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        # --- 根据运行模式执行不同逻辑 ---
        if trial:  # Optuna 模式
            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
        else:  # 最终训练模式
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), config['model_save_path'])
                if not trial:
                    print(f"Validation loss improved. Best model saved to {config['model_save_path']}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['early_stopping_patience']:
                    print(f"Early stopping triggered at Epoch {epoch_offset + epoch + 1}.")
                    break
                    
    return best_val_loss, train_losses, val_losses


def evaluate_model(model, data_loader, output_feature_scaler, config, output_csv_path=None):
    """
    评估模型，并可选择将结果导出为 CSV 文件。
    - 修复了时间戳处理的Bug：现在可以正确处理 `TOCDataset` 和 `PrecomputedTOCDataset`
      中的时间戳，确保CSV文件中的时间戳与预测值一一对应。
    """
    model.eval()
    all_predictions, all_true_values = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(config['device'])
            inputs = adjust_input_for_model(inputs, model)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_true_values.extend(targets.cpu().numpy())

    # 反归一化并计算性能指标
    all_predictions_denorm = output_feature_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
    all_true_values_denorm = output_feature_scaler.inverse_transform(np.array(all_true_values).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(all_true_values_denorm, all_predictions_denorm))
    mae = mean_absolute_error(all_true_values_denorm, all_predictions_denorm)
    r2 = r2_score(all_true_values_denorm, all_predictions_denorm)
    print(f"Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")

    # --- 导出结果到 CSV 文件 ---
    if output_csv_path:
        try:
            timestamps = data_loader.dataset.timestamps
            
            # 对于使用TOCDataset的测试集，其原始时间戳长度可能不等于样本数，需要修正。
            if len(timestamps) != len(all_true_values_denorm):
                print(f"Warning: Timestamps array length ({len(timestamps)}) mismatches results length ({len(all_true_values_denorm)}). Attempting to fix...")
                # 这种情况通常发生在测试集（TOCDataset），其timestamps属性是完整的Series
                if isinstance(data_loader.dataset, TOCDataset):
                    seq_len = config['sequence_length']
                    horizon = config['forecast_horizon']
                    # 截取时间戳以匹配样本的起始点
                    timestamps = timestamps.iloc[seq_len + horizon - 1:].reset_index(drop=True)

            if len(timestamps) == len(all_true_values_denorm):
                results_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'true_value': all_true_values_denorm,
                    'predicted_value': all_predictions_denorm
                })
                output_dir = os.path.dirname(output_csv_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                results_df.to_csv(output_csv_path, index=False)
                print(f"Evaluation results successfully saved to: {output_csv_path}")
            else:
                 print(f"Error: Could not align timestamps with results. CSV file will not be saved.")

        except AttributeError:
            print("Warning: 'timestamps' attribute not found in dataset. Cannot save results to CSV.")
        except Exception as e:
            print(f"An error occurred while saving CSV: {e}")

    return all_predictions_denorm, all_true_values_denorm, rmse, mae, r2


def run_optuna_objective(trial, train_dataset, validation_dataset, model_name, input_size, current_config):
    """
    Optuna 的目标函数，直接调用重构后的 `train_model` 函数。
    """
    # --- 1. 定义超参数搜索空间 ---
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    loss_weight_factor = trial.suggest_float('loss_weight_factor', 1.5, 15.0)

    model_params = {}
    if model_name == 'lstm':
        model_params['hidden_size'] = trial.suggest_categorical('hidden_size', [64, 128, 256])
        model_params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        model_params['dropout'] = trial.suggest_float('dropout', 0.2, 0.5)
        if model_params['num_layers'] == 1: model_params['dropout'] = 0.0
    
    elif model_name == 'tcn':
        num_levels = trial.suggest_int('num_levels', 2, 4)
        num_channels_per_level = trial.suggest_categorical('num_channels_per_level', [24, 32, 48])
        model_params['num_channels'] = [num_channels_per_level] * num_levels
        model_params['kernel_size'] = trial.suggest_categorical('kernel_size', [2, 3, 5])
        model_params['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
        model_params['output_size'] = 1

    elif model_name == 'tcn_lstm_attention':
        model_params['tcn_num_channels'] = trial.suggest_categorical('tcn_num_channels', [[25, 50], [32, 64]])
        model_params['tcn_kernel_size'] = trial.suggest_categorical('tcn_kernel_size', [2, 3])
        model_params['tcn_dropout'] = trial.suggest_float('tcn_dropout', 0.1, 0.4)
        model_params['lstm_hidden_size'] = trial.suggest_categorical('lstm_hidden_size', [64, 128])
        model_params['lstm_num_layers'] = trial.suggest_int('lstm_num_layers', 1, 2)
        model_params['lstm_dropout'] = trial.suggest_float('lstm_dropout', 0.2, 0.5)
        if model_params['lstm_num_layers'] == 1: model_params['lstm_dropout'] = 0.0
    
    else:
        raise ValueError(f"Unknown model name '{model_name}' for hyperparameter tuning.")

    # --- 2. 实例化模型、数据加载器、损失函数和优化器 ---
    ModelClass = MODELS[model_name]
    model_params['input_size'] = input_size
    model = ModelClass(**model_params).to(current_config['device'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


    criterion = WeightedMSELoss(
        warning_threshold=current_config['scaled_warning_threshold'],
        weight_factor=loss_weight_factor
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- 3. 调用通用的训练函数进行训练和评估 ---
    # 将 trial 对象传递给 train_model，以启用 Optuna 模式
    best_val_loss, _, _ = train_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        current_config,
        trial=trial  # 传入trial对象
    )

    return best_val_loss

plot.py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from src.utils.helpers import adjust_input_for_model


def plot_loss_curves(train_losses, val_losses, results_dir, title_suffix=""):
    """
    绘制并保存训练和验证损失曲线图。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # 自动生成安全的文件名
    filename_suffix = "".join(c for c in title_suffix if c.isalnum()).lower()
    plt.savefig(os.path.join(results_dir, f'loss_curve_{filename_suffix}.png'))
    plt.close() # 使用 plt.close() 以便在后台脚本中无头运行

def plot_actual_vs_predicted(actual_values, predicted_values, warning_threshold, train_split_idx, results_dir, title_suffix=""):
    """
    绘制并保存真实值与预测值的对比图。
    """
    plt.figure(figsize=(18, 9))
    plt.plot(actual_values, label='Actual TOC', color='#7995c4', alpha=0.7)
    plt.plot(predicted_values, label='Predicted TOC', color='#c44e52', linestyle='--', alpha=0.7)
    plt.axhline(y=warning_threshold, color='green', linestyle=':', label=f'Warning Threshold ({warning_threshold})')
    plt.axvline(x=train_split_idx, color='purple', linestyle='--', label='Train/Test Split')
    plt.title(f'Actual vs. Predicted TOC {title_suffix}')
    plt.xlabel('Time Step')
    plt.ylabel('TOC Value')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    filename_suffix = "".join(c for c in title_suffix if c.isalnum()).lower()
    plt.savefig(os.path.join(results_dir, f'model_training_{filename_suffix}.png'))
    plt.close()

# def plot_warning_points(actual_values, predicted_values, warning_threshold, train_split_idx, results_dir, title_suffix=""):
#     """
#     绘制并高亮显示预警点。
#     """
#     predicted_warnings = predicted_values >= warning_threshold
#     true_warnings = actual_values >= warning_threshold

#     plt.figure(figsize=(18, 9))
#     plt.plot(actual_values, label='Actual TOC', color='blue', alpha=0.7)
#     plt.plot(predicted_values, label='Predicted TOC', color='red', linestyle='--', alpha=0.7)
    
#     true_warning_indices = np.where(true_warnings)[0]
#     plt.scatter(true_warning_indices, actual_values[true_warning_indices],
#                 color='green', marker='o', s=50, label='Actual Warning (TP + FN)', zorder=5)
    
#     predicted_warning_indices = np.where(predicted_warnings)[0]
#     plt.scatter(predicted_warning_indices, predicted_values[predicted_warning_indices],
#                 color='orange', marker='x', s=50, label='Predicted Warning (TP + FP)', zorder=5)
    
#     plt.axhline(y=warning_threshold, color='green', linestyle=':', label=f'Warning Threshold ({warning_threshold})')
#     plt.axvline(x=train_split_idx, color='purple', linestyle='--', label='Train/Test Split')
#     plt.title(f'Highlighted Warning Points {title_suffix}')
#     plt.xlabel('Time Step')
#     plt.ylabel('TOC Value')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     filename_suffix = "".join(c for c in title_suffix if c.isalnum()).lower()
#     plt.savefig(os.path.join(results_dir, f'warning_points_{filename_suffix}.png'))
#     plt.close()

# def simulate_realtime_prediction(model, scaler, output_feature_scaler, config, test_df, input_feature_indices):
#     """
#     模拟实时预警流程并绘图。
#     """
#     print("\n[功能] 实时预警模拟...")
#     num_simulated_steps = 50
#     if len(test_df) < num_simulated_steps + config['sequence_length']:
#         print("测试数据不足，无法进行实时模拟。")
#         return
        
#     simulated_raw_data_df = test_df.iloc[-num_simulated_steps - config['sequence_length'] + 1:]
#     simulated_scaled_data = scaler.transform(simulated_raw_data_df)

#     simulated_predictions, simulated_warnings = [], []
#     model.eval()

#     for i in range(num_simulated_steps):
#         current_input_sequence_scaled = simulated_scaled_data[i : i + config['sequence_length'], input_feature_indices].T
#         input_tensor = torch.tensor(current_input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(config['device'])
        
#         with torch.no_grad():
#             # ## [MODIFIED] ##
#             # 此处调用的是从共享模块导入的函数
#             input_tensor = adjust_input_for_model(input_tensor, model)
#             predicted_toc_scaled = model(input_tensor).cpu().item()
        
#         predicted_toc_denorm = output_feature_scaler.inverse_transform(np.array([[predicted_toc_scaled]])).flatten()[0]
#         simulated_predictions.append(predicted_toc_denorm)

#         if predicted_toc_denorm >= config['warning_threshold']:
#             warning_status = "!!! WARNING !!!"
#             simulated_warnings.append(1)
#         else:
#             warning_status = "Normal"
#             simulated_warnings.append(0)
            
#         print(f"  模拟步 {i+1:02d}: 预测TOC = {predicted_toc_denorm:.4f}, 状态: {warning_status}")

#     # --- 绘图部分 ---
#     plt.figure(figsize=(14, 7))
#     plt.plot(simulated_predictions, label='Simulated Predicted TOC', color='red', linestyle='--')
#     plt.axhline(y=config['warning_threshold'], color='green', linestyle=':', label=f'Warning Threshold ({config["warning_threshold"]})')
#     simulated_warning_indices = np.where(np.array(simulated_warnings) == 1)[0]
#     plt.scatter(simulated_warning_indices, np.array(simulated_predictions)[simulated_warning_indices],
#                 color='purple', marker='^', s=70, label='Predicted Warning Event', zorder=5)
#     plt.title('Simulated Real-time TOC Prediction and Warning')
#     plt.xlabel('Simulated Time Step')
#     plt.ylabel('TOC Value')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(config.get('results_dir', '.'), 'simulated_realtime_prediction.png'))
#     plt.close()


def calculate_warning_metrics(all_true_values_denorm, all_predictions_denorm, warning_threshold):
    """
    计算并打印预警相关的分类指标 (TP, FP, FN, TN, F1-Score等)。
    """
    predicted_warnings = all_predictions_denorm >= warning_threshold
    true_warnings = all_true_values_denorm >= warning_threshold

    TP = np.sum(predicted_warnings & true_warnings)
    FP = np.sum(predicted_warnings & ~true_warnings)
    FN = np.sum(~predicted_warnings & true_warnings)
    TN = np.sum(~predicted_warnings & ~true_warnings)

    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- 预警系统性能评估 ---")
    print(f"预警阈值: {warning_threshold}")
    print(f"  - TP (正确预警): {TP}")
    print(f"  - FP (错误预警/误报): {FP}")
    print(f"  - FN (遗漏预警/漏报): {FN}")
    print(f"  - TN (正确无预警): {TN}")
    print(f"  - 准确率 (Accuracy): {accuracy:.4f}")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - 召回率 (Recall): {recall:.4f}")
    print(f"  - F1 Score: {f1_score:.4f}")
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1_score}

helper.py
import torch

def adjust_input_for_model(inputs, model):
    """
    一个健壮的共享辅助函数，根据模型定义的 input_format 调整数据维度。
    - 'NCL': Batch, Channels, Length (TCN 的需求)
    - 'NLC': Batch, Length, Channels (LSTM 的需求)
    假设数据加载器默认输出 'NCL' [B, C, L] 格式。
    """
    required_format = getattr(model, 'input_format', 'NCL')
    
    # 如果模型需要 NLC (例如 LSTM)，则进行维度转换
    if required_format == 'NLC':
        # 从 [B, C, L] 转换为 [B, L, C]
        return inputs.permute(0, 2, 1)
        
    # 如果模型需要 NCL (例如 TCN)，则不需要改变
    return inputs