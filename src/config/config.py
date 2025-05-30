import torch
import os
CONFIG = {
    'data_path': '/home/hmd24/project/TOC_Warning_System/data/processed/TOC_dataset.xlsx',
    'output_feature': 'Endpoint',
    'input_features': ['BD', 'PW', 'DI', 'RO', 'MD'],
    'sequence_length': 24,
    'forecast_horizon': 1,
    'train_ratio': 0.8,
    'batch_size': 64, 
    'num_epochs': 200,
    'learning_rate': 0.001, 
    'early_stopping_patience': 20, 
    'warning_threshold': 0.65,
    'model_save_path': '../checkpoints/best_tcn_lstm_attention_toc_model.pth',
    'results_dir': '../results/plots/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Scaler
    'scaler_type': 'Standard', 
    
    # L2 正则化强度
    'weight_decay': 1e-5, 

    # TCN specific parameters 
    'tcn_num_channels': [30, 30, 30], 
    'tcn_kernel_size': 3,             
    'tcn_dropout': 0.3,               
    
    # LSTM specific parameters 
    'lstm_hidden_size': 64,          
    'lstm_num_layers': 2,            
    'lstm_dropout': 0.3,             

    # Optuna specific parameters
    'n_trials': 50, # Optuna 试验次数
    'timeout': 7200 # Optuna 运行时间上限 (秒)
}
# 确保路径是相对项目根目录的
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 当前文件 (config.py) 的目录
PROJECT_ROOT = os.path.dirname(BASE_DIR) # 项目根目录

# 调整所有路径为相对于项目根目录
CONFIG['data_path'] = os.path.join(PROJECT_ROOT, CONFIG['data_path'])
CONFIG['model_save_path'] = os.path.join(PROJECT_ROOT, CONFIG['model_save_path'])
CONFIG['results_dir'] = os.path.join(PROJECT_ROOT, CONFIG['results_dir'])

# 确保结果目录存在
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)