import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

class TOCDataset(Dataset):
    def __init__(self, data, input_features_idx, output_feature_idx, sequence_length, forecast_horizon):
        self.data = data
        self.input_features_idx = input_features_idx
        self.output_feature_idx = output_feature_idx
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # 这里的 input_seq 需要是 (num_features, sequence_length)
        input_seq = self.data[idx : idx + self.sequence_length, self.input_features_idx].T
        target_toc = self.data[idx + self.sequence_length + self.forecast_horizon - 1, self.output_feature_idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_toc, dtype=torch.float32)

def load_and_preprocess_data(config):
    df = pd.read_excel(config['data_path'])
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        print("数据已根据 'timestamp' 列进行时序排序。")
    else:
        print("警告: 未找到 'timestamp' 列，请确保数据已按时间顺序排列。")

    features_df = df[config['input_features'] + [config['output_feature']]]
    train_size = int(len(features_df) * config['train_ratio'])
    train_df = features_df.iloc[:train_size]
    test_df = features_df.iloc[train_size:]

    if config['scaler_type'] == 'MinMax':
        scaler = MinMaxScaler()
        print("使用 MinMaxScaler 进行数据归一化。")
    elif config['scaler_type'] == 'Standard':
        scaler = StandardScaler()
        print("使用 StandardScaler 进行数据标准化。")
    else:
        raise ValueError("scaler_type 必须是 'MinMax' 或 'Standard'")

    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)

    # 专门为 output_feature 创建 scaler，用于反归一化和阈值转换
    output_feature_scaler = None
    output_feature_scaler = StandardScaler() if config['scaler_type'] == 'Standard' else MinMaxScaler()
    output_feature_scaler.fit(train_df[[config['output_feature']]])

    # 计算归一化后的预警阈值并更新到 config
    config['scaled_warning_threshold'] = output_feature_scaler.transform(
        np.array([[config['warning_threshold']]])
    ).flatten()[0]
    print(f"归一化后的预警阈值: {config['scaled_warning_threshold']:.4f}")

    print(f"训练集形状: {train_scaled.shape}")
    print(f"测试集形状: {test_scaled.shape}")

    # 获取输入和输出特征在 `features_df` 中的索引
    input_feature_indices = [features_df.columns.get_loc(col) for col in config['input_features']]
    output_feature_index = features_df.columns.get_loc(config['output_feature'])

    train_dataset = TOCDataset(train_scaled, input_feature_indices, output_feature_index, config['sequence_length'], config['forecast_horizon'])
    test_dataset = TOCDataset(test_scaled, input_feature_indices, output_feature_index, config['sequence_length'], config['forecast_horizon'])

    return train_dataset, test_dataset, output_feature_scaler, len(config['input_features']), config # 返回更新后的config