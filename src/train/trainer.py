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