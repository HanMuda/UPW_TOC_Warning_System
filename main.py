# main.py

import torch
import torch.nn as nn
import optuna
import os
import pandas as pd
import numpy as np

from src.config.config import CONFIG
from src.dataset.data_processor import load_and_preprocess_data, TOCDataset, DataLoader
from src.model.model_TCN_LSTM_Attention import TOCPredictor_TCN_LSTM_Attention
from src.training.trainer import run_optuna_objective, train_model, evaluate_model
from src.visualization.plot import plot_loss_curves, plot_actual_vs_predicted, plot_warning_points, simulate_realtime_prediction, calculate_warning_metrics

# --- 0. 全局设置 ---
print(f"Using device: {CONFIG['device']}")

# --- 1. 数据加载与预处理 ---
print("\n--- 加载并预处理数据 ---")
train_dataset, test_dataset, output_feature_scaler, input_size_for_model, updated_config = load_and_preprocess_data(CONFIG)
# 更新全局CONFIG with scaled_warning_threshold
CONFIG.update(updated_config) 

# 为了 Optuna 目标函数能够访问这些数据集
global_train_dataset = train_dataset
global_test_dataset = test_dataset
global_input_size_for_model = input_size_for_model

# --- 2. 运行超参数调优 ---
print("\n--- 开始超参数调优 ---")
# 使用 lambda 表达式将固定参数传入 run_optuna_objective
objective_wrapper = lambda trial: run_optuna_objective(
    trial, 
    global_train_dataset, 
    global_test_dataset, 
    global_input_size_for_model, 
    CONFIG
)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective_wrapper, n_trials=CONFIG['n_trials'], timeout=CONFIG['timeout'])

print("\n--- 超参数调优完成 ---")
print(f"最佳试验的超参数: {study.best_trial.params}")
print(f"最佳试验的验证损失: {study.best_trial.value:.6f}")

# 可以保存最佳试验结果
study.trials_dataframe().to_csv(os.path.join(os.path.dirname(CONFIG['results_dir']), 'hyperparameter_tuning_results.csv'))

# --- 3. 使用最佳超参数训练最终模型 ---
print("\n--- 使用最佳超参数训练最终模型 ---")

# 从 Optuna 结果中获取最佳超参数并更新 CONFIG
CONFIG.update(study.best_trial.params)

# 确保 LSTM dropout 在单层时为 0
if CONFIG['lstm_num_layers'] == 1:
    CONFIG['lstm_dropout'] = 0.0

# 重新创建 DataLoader (使用最佳 batch_size)
train_loader_final = DataLoader(global_train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
test_loader_final = DataLoader(global_test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# 实例化最终模型
final_model = TOCPredictor_TCN_LSTM_Attention(
    input_size=global_input_size_for_model,
    tcn_num_channels=CONFIG['tcn_num_channels'],
    tcn_kernel_size=CONFIG['tcn_kernel_size'],
    tcn_dropout=CONFIG['tcn_dropout'],
    lstm_hidden_size=CONFIG['lstm_hidden_size'],
    lstm_num_layers=CONFIG['lstm_num_layers'],
    lstm_dropout=CONFIG['lstm_dropout']
).to(CONFIG['device'])

# 定义损失函数和优化器 (使用自定义损失函数和最佳参数)
criterion = nn.MSELoss()
final_optimizer = torch.optim.Adam(
    final_model.parameters(), 
    lr=CONFIG['learning_rate'], 
    weight_decay=CONFIG['weight_decay']
)

# 训练最终模型
best_val_loss_final, train_losses_final, val_losses_final = train_model(
    final_model, 
    train_loader_final, 
    test_loader_final, 
    criterion, 
    final_optimizer, 
    CONFIG,
    epoch_offset=0 # 初始训练
)
print("\n--- 最终模型训练完成 ---")

# --- 4. 评估最终模型 ---
print("\n--- 最终模型评估结果 (在测试集上) ---")
# 加载最佳模型权重
final_model.load_state_dict(torch.load(CONFIG['model_save_path']))
all_predictions_test_denorm, all_true_values_test_denorm, rmse, mae, r2 = evaluate_model(
    final_model, test_loader_final, output_feature_scaler, CONFIG
)
calculate_warning_metrics(all_true_values_test_denorm, all_predictions_test_denorm, CONFIG['warning_threshold'])

# --- 5. 可视化结果 ---
print("\n--- 可视化结果 ---")
plot_loss_curves(train_losses_final, val_losses_final, CONFIG['results_dir'], title_suffix="(Final Model)")

# 准备全数据集的预测数据
full_scaled_data = np.vstack((train_dataset.data, test_dataset.data))
full_dataset = TOCDataset(full_scaled_data, train_dataset.input_features_idx, train_dataset.output_feature_idx, CONFIG['sequence_length'], CONFIG['forecast_horizon'])
full_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

all_predictions_full_denorm, all_true_values_full_denorm, _, _, _ = evaluate_model(
    final_model, full_loader, output_feature_scaler, CONFIG
)

plot_actual_vs_predicted(
    all_true_values_full_denorm, 
    all_predictions_full_denorm, 
    CONFIG['warning_threshold'], 
    len(train_dataset), # train_split_idx
    CONFIG['results_dir'], 
    title_suffix="(All Data, Final Model)"
)

plot_warning_points(
    all_true_values_full_denorm, 
    all_predictions_full_denorm, 
    CONFIG['warning_threshold'], 
    len(train_dataset), # train_split_idx
    CONFIG['results_dir'], 
    title_suffix="(All Data, Final Model)"
)