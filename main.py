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
    calculate_warning_metrics, 
    plot_confusion_matrix,
    plot_prediction_error
)
from src.utils.helpers import set_seeds


def main():
    # ==========================================================
    set_seeds(seed=42) 
    # ==========================================================
    # ===== 0. 全局配置 =====
    try:
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("错误: config.yaml 文件未找到。请确保它在 'src/config/' 目录下。")
        return

    # 提取当前运行的配置
    run_config = config['run_config']
    model_name = run_config['model_name']
    run_config['model_save_path'] = run_config['model_save_path'].format(model_name=model_name)
    
    print(f"--- 使用配置文件: src/config/config.yaml ---")
    print(f"--- 运行的模型: {model_name} ---")
    print(f"--- 模型将保存至: {run_config['model_save_path']} ---")
    print(f"--- 使用设备: {run_config['device']} ---")

    # ===== 1. 数据加载与预处理 =====
    print("\n[步骤 1/5] 加载并预处理数据...")
    full_timestamps, train_dataset, val_dataset, test_dataset, output_feature_scaler, input_size, updated_config, train_scaled_original, val_scaled_original, test_scaled_original = load_and_preprocess_data(run_config)
    run_config.update(updated_config) 

    best_params = {} # 先初始化一个空字典
    
    if run_config.get('perform_tuning', True):
        # ===== 2. 超参数调优 (Optuna) =====
        print(f"\n[步骤 2/5] 开关已开启，开始为 '{model_name}' 模型进行超参数调优...")
        
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
    
    else: 
        print(f"\n[步骤 2/5] 开关已关闭，跳过超参数调优。")
        print(f"--- 将直接使用 config.yaml 中 'model_params.{model_name}' 下定义的参数 ---")
        best_params = config['model_params'][model_name]
        print(f"使用的预设超参数: {best_params}")

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
    test_csv_filename = f"{model_name}_test_data_{csv_base_name}"
    test_csv_path = os.path.join(results_dir, test_csv_filename)
    print(f"测试集评估结果将保存到: {test_csv_path}")
    
    all_predictions_test_denorm, all_true_values_test_denorm, _, _, _ = evaluate_model(
        final_model, 
        test_loader_final, 
        output_feature_scaler, 
        run_config,
        output_csv_path=test_csv_path 
    )

    calculate_warning_metrics(all_true_values_test_denorm, all_predictions_test_denorm, run_config['warning_threshold'])
    plot_confusion_matrix(all_true_values_test_denorm, all_predictions_test_denorm, 
                          run_config['warning_threshold'], results_dir, title_suffix=f"({model_name} Test Set)")

    # ===== 5. 可视化 =====
    print("\n[步骤 5/5] 生成可视化结果...")
    plot_loss_curves(train_losses_final, val_losses_final, results_dir, title_suffix=f"({model_name} )")

    # 为了在完整数据集上绘图，重新准备 full_loader
    full_scaled_data = np.vstack((train_scaled_original, val_scaled_original, test_scaled_original))
    from src.data_loader.data_processor import TOCDataset as FullTOCDataset # 避免混淆
        
    full_dataset = TOCDataset(full_scaled_data,
                            full_timestamps, 
                            val_dataset.input_features_idx, 
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

    # 注意：这里的长度是可生成样本的数量，而不是原始数据点的数量
    seq_len = run_config['sequence_length']
    horizon = run_config['forecast_horizon']
    
    # 第一个分割点：训练集样本数
    train_val_split_idx = len(train_scaled_original) - seq_len - horizon + 1
    # 第二个分割点：训练集样本数 + 验证集样本数
    val_test_split_idx = train_val_split_idx + (len(val_scaled_original) - seq_len - horizon + 1)

    start_index = seq_len + horizon - 1
    timestamps_for_plot = full_timestamps.iloc[start_index:].reset_index(drop=True)

    plot_actual_vs_predicted(
        actual_values=all_true_values_full_denorm, 
        predicted_values=all_predictions_full_denorm, 
        timestamps_for_plot=timestamps_for_plot,  # 传入时间戳
        warning_threshold=run_config['warning_threshold'], 
        train_val_split_idx=train_val_split_idx, 
        val_test_split_idx=val_test_split_idx, 
        results_dir=results_dir, 
        title_suffix=f"({model_name} _all_data)"
    )
    
    plot_prediction_error(
        actual_values=all_true_values_full_denorm,
        predicted_values=all_predictions_full_denorm,
        timestamps_for_plot=timestamps_for_plot,
        results_dir=results_dir,
        title_suffix=f"({model_name} _all_data)"
    )

    print(f"\n程序运行结束。所有结果已保存到 '{results_dir}' 目录。")


if __name__ == '__main__':
    main()