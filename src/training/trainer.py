import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import os

from src.model.model_TCN_LSTM_Attention import TOCPredictor_TCN_LSTM_Attention, WeightedMSELoss
from src.dataset.data_processor import TOCDataset # 需要导入以在objective中创建 DataLoader
from src.config.config import CONFIG

def train_model(model, train_loader, val_loader, criterion, optimizer, config, epoch_offset=0):
    """
    训练模型的通用函数。
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch_offset + epoch + 1}/{config['num_epochs']}], Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # 仅在最终训练时保存模型
            if epoch_offset == 0: # 标记为是最终训练
                torch.save(model.state_dict(), config['model_save_path'])
                print(f"--- 最佳模型保存到 {config['model_save_path']} ---")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['early_stopping_patience']:
                print(f"早停条件达到，在 Epoch {epoch_offset + epoch + 1} 停止训练。")
                break
    return best_val_loss, train_losses, val_losses

def evaluate_model(model, data_loader, output_feature_scaler, config):
    """
    评估模型并返回预测值和真实值 (反归一化)。
    """
    model.eval()
    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_true_values.extend(targets.cpu().numpy())

    all_predictions_denorm = output_feature_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
    all_true_values_denorm = output_feature_scaler.inverse_transform(np.array(all_true_values).reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(all_true_values_denorm, all_predictions_denorm))
    mae = mean_absolute_error(all_true_values_denorm, all_predictions_denorm)
    r2 = r2_score(all_true_values_denorm, all_predictions_denorm)

    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

    return all_predictions_denorm, all_true_values_denorm, rmse, mae, r2

def run_optuna_objective(trial, train_dataset, test_dataset, input_size_for_model, current_config):
    """
    Optuna 的目标函数。
    """
    # 定义超参数搜索空间
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    loss_weight_factor = trial.suggest_float('loss_weight_factor', 1.5, 5.0)
    
    tcn_num_channels = trial.suggest_categorical('tcn_num_channels', [[20, 20, 20], [30, 30, 30], [40, 40, 40], [30, 60, 120]])
    tcn_kernel_size = trial.suggest_categorical('tcn_kernel_size', [2, 3, 4])
    tcn_dropout = trial.suggest_float('tcn_dropout', 0.2, 0.5)
    
    lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [32, 64, 128])
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.2, 0.5)
    
    if lstm_num_layers == 1:
        lstm_dropout = 0.0

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = TOCPredictor_TCN_LSTM_Attention(
        input_size=input_size_for_model,
        tcn_num_channels=tcn_num_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout
    ).to(current_config['device'])

    # 定义损失函数和优化器
    criterion = WeightedMSELoss(
        warning_threshold=current_config['scaled_warning_threshold'], # 使用归一化后的阈值
        weight_factor=loss_weight_factor
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(current_config['num_epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(current_config['device']), targets.to(current_config['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(current_config['device']), targets.to(current_config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_running_loss / len(test_dataset)
        
        trial.report(epoch_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= current_config['early_stopping_patience']:
                break
    
    return best_val_loss