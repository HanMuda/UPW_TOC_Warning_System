import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.dates as mdates

from src.utils.helpers import adjust_input_for_model

# ===== 设置Matplotlib支持中文显示 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Heiti TC', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 
# ====================================


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

def plot_actual_vs_predicted(actual_values, predicted_values, timestamps_for_plot, 
                             warning_threshold, 
                             train_val_split_idx, val_test_split_idx,
                             results_dir, title_suffix=""):
    """
    绘制并保存真实值与预测值的对比图，横轴为日期。
    """
    fig, ax = plt.subplots(figsize=(18, 9))
    
    # 使用时间戳作为横轴数据
    ax.plot(timestamps_for_plot, actual_values, label='真实值 (Actual TOC)', color='#7995c4', alpha=0.7)
    ax.plot(timestamps_for_plot, predicted_values, label='预测值 (Predicted TOC)', color='#c44e52', linestyle='--', alpha=0.7)
    
    ax.axhline(y=warning_threshold, color='green', linestyle=':', label=f'预警阈值 ({warning_threshold})')
    
    # 使用时间戳索引来定位分割线的位置
    ax.axvline(x=timestamps_for_plot.iloc[train_val_split_idx], color='purple', linestyle='--', label='训练/验证 分割点')
    ax.axvline(x=timestamps_for_plot.iloc[val_test_split_idx], color='orange', linestyle='--', label='验证/测试 分割点')
    
    # 更新坐标轴标签
    ax.set_xlabel('日期 (Date)')
    ax.set_ylabel('UPW TOC')
    ax.legend() 
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.autofmt_xdate()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    filename_suffix = "".join(c for c in title_suffix if c.isalnum()).lower()
    plt.savefig(os.path.join(results_dir, f'model_training_{filename_suffix}.png'))
    plt.close()

# src/visualization/plot.py

# ... (文件顶部的 import 和其他函数保持不变) ...

# [新增] 绘制预测误差时间序列图的函数
def plot_prediction_error(actual_values, predicted_values, timestamps_for_plot, 
                          results_dir, title_suffix=""):
    """
    计算并绘制预测误差（真实值 - 预测值）的时间序列图。
    """
    # 计算误差
    error = actual_values - predicted_values
    
    # 开始绘图
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # 绘制误差曲线
    ax.plot(timestamps_for_plot, error, label='预测误差 (Prediction Error)', color='#2ca02c', alpha=0.8)
    
    # 绘制一条 y=0 的参考线，方便观察误差的正负
    ax.axhline(y=0, color='red', linestyle='--', label='零误差参考线 (Zero Error Ref)')
    
    # 设置图表标题和坐标轴标签
    ax.set_title(f'预测误差时间序列图 {title_suffix}')
    ax.set_xlabel('日期 (Date)')
    ax.set_ylabel('预测误差 (Actual - Predicted)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 自动格式化X轴的日期标签
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # 保存图表
    filename_suffix = "".join(c for c in title_suffix if c.isalnum()).lower()
    save_path = os.path.join(results_dir, f'prediction_error_{filename_suffix}.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"预测误差图已保存到: {save_path}")

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

    print("\n--- 预警系统性能评估 (测试集) ---")
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

def plot_confusion_matrix(y_true, y_pred, threshold, results_dir, title_suffix=""):
    """
    计算并绘制混淆矩阵图。
    """
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常 (Normal)', '预警 (Warning)'], 
                yticklabels=['正常 (Normal)', '预警 (Warning)'])
    plt.title(f'混淆矩阵 (Confusion Matrix) {title_suffix}')
    plt.xlabel('预测标签 (Predicted Label)')
    plt.ylabel('真实标签 (True Label)')
    
    filename_suffix = "".join(c for c in title_suffix if c.isalnum()).lower()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{filename_suffix}.png'))
    plt.close()
    print(f"混淆矩阵图已保存到: {os.path.join(results_dir, f'confusion_matrix_{filename_suffix}.png')}")
