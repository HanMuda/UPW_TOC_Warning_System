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
#  特征工程
# ===================================================================
def create_features(df, config):
    """
    从原始DataFrame创建所有可能的衍生特征。
    它使用 config 中的 'base_features' 列表来决定对哪些特征进行衍生。
    """
    print("开始执行特征工程，创建所有衍生特征...")
    
    # --- 准备工作 ---
    base_features = config.get('base_features')
    output_feature = config.get('output_feature')
    hrt_map = config.get('HRT_values')

    if not base_features:
        raise ValueError("错误: 'base_features' 列表未在 config.yaml 中定义。")
    if not isinstance(hrt_map, dict):
        raise TypeError("错误: 'HRT_values' 必须在 config.yaml 中定义为一个字典。")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- 1. 创建时间特征 ---
    print("  - 步骤 1/2: 创建时间周期特征...")
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df = df.drop(columns=['hour', 'dayofweek', 'month'])
    
    # --- 2. 创建差值特征 & 3. 创建变化率特征 ---
    # 在一个循环中完成，更高效
    print("  - 步骤 2/2: 创建差值和变化率特征...")
    for feature in base_features:
        # 创建差值特征
        diff_feature_name = f'{feature}_minus_{output_feature}'
        df[diff_feature_name] = df[feature] - df[output_feature]
        
        # 创建变化率特征
        hrt_value = hrt_map.get(feature)
        if hrt_value is None:
            raise ValueError(f"错误: 特征 '{feature}' 的HRT值未在 'HRT_values' 字典中找到。")
        rate_feature_name = f'{feature}_rate_of_change'
        df[rate_feature_name] = df[diff_feature_name] / hrt_value
        
    print("特征工程创建完成。")
    return df
# ===================================================================
#  数据加载与预处理主函数
# ===================================================================

def load_and_preprocess_data(config):
    """
    加载、预处理、增强并准备数据集的主函数。
    """
    # --- 1. 加载和排序 ---
    df = pd.read_csv(config['data_path'])

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        print("数据已根据 'timestamp' 列进行时序排序。")
    else:
        print("警告: 未找到 'timestamp' 列，请确保数据已按时间顺序排列。")

    df = create_features(df, config)

    # 分离出特征和时间戳
    timestamps = df['timestamp']

    # 提取特征
    final_model_inputs = config['final_model_inputs'] 
    output_feature = config['output_feature']

    features_df = df[final_model_inputs + [output_feature]] 
    print(f"最终用于模型的输入特征共 {len(final_model_inputs)} 个: {final_model_inputs}")

    # --- 2. 数据集划分 ---
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    
    train_size = int(len(features_df) * train_ratio)
    val_size = int(len(features_df) * val_ratio)
    
    train_df = features_df.iloc[:train_size]
    val_df = features_df.iloc[train_size : train_size + val_size]
    test_df = features_df.iloc[train_size + val_size:]

    # 分割时间戳
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
    input_feature_indices = [features_df.columns.get_loc(col) for col in final_model_inputs]
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

    input_size = len(final_model_inputs)
    updated_config = config
    return timestamps, train_dataset, val_dataset, test_dataset, output_feature_scaler, input_size, updated_config, train_scaled, val_scaled, test_scaled
