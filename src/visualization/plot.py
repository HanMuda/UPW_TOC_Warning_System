# src/utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

results_dir=[]

def plot_loss_curves(train_losses, val_losses, results_dir, title_suffix=""):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'training_validation_loss_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()

def plot_actual_vs_predicted(actual_values, predicted_values, warning_threshold, train_split_idx, results_dir, title_suffix=""):
    plt.figure(figsize=(18, 9))
    plt.plot(actual_values, label='Actual TOC', color='blue', alpha=0.7)
    plt.plot(predicted_values, label='Predicted TOC', color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=warning_threshold, color='green', linestyle=':', label=f'Warning Threshold ({warning_threshold})')
    plt.axvline(x=train_split_idx, color='purple', linestyle='--', label='Train/Test Split')
    plt.title(f'Actual vs. Predicted TOC with Warning Threshold {title_suffix}')
    plt.xlabel('Time Step')
    plt.ylabel('TOC Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'actual_vs_predicted_toc_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()

def plot_warning_points(actual_values, predicted_values, warning_threshold, train_split_idx, results_dir, title_suffix=""):
    predicted_warnings = predicted_values >= warning_threshold
    true_warnings = actual_values >= warning_threshold

    plt.figure(figsize=(18, 9))
    plt.plot(actual_values, label='Actual TOC', color='blue', alpha=0.7)
    plt.plot(predicted_values, label='Predicted TOC', color='red', linestyle='--', alpha=0.7)
    
    true_warning_indices = np.where(true_warnings)[0]
    plt.scatter(true_warning_indices, actual_values[true_warning_indices],
                color='green', marker='o', s=50, label='Actual Warning (TP + FN)')
    
    predicted_warning_indices = np.where(predicted_warnings)[0]
    plt.scatter(predicted_warning_indices, predicted_values[predicted_warning_indices],
                color='orange', marker='x', s=50, label='Predicted Warning (TP + FP)')
    
    plt.axhline(y=warning_threshold, color='green', linestyle=':', label=f'Warning Threshold ({warning_threshold})')
    plt.axvline(x=train_split_idx, color='purple', linestyle='--', label='Train/Test Split')
    plt.title(f'Actual vs. Predicted TOC with Highlighted Warning Points {title_suffix}')
    plt.xlabel('Time Step')
    plt.ylabel('TOC Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'warning_points_highlighted_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()

def simulate_realtime_prediction(model, scaler, output_feature_scaler, config, test_df, input_feature_indices):
    print("\n--- 实时预警模拟 ---")
    num_simulated_steps = 50
    simulated_raw_data_df = test_df[-num_simulated_steps - config['sequence_length'] + 1 : ]
    simulated_scaled_data = scaler.transform(simulated_raw_data_df)

    simulated_predictions = []
    simulated_warnings = []

    model.eval()

    for i in range(num_simulated_steps):
        current_input_sequence_scaled = simulated_scaled_data[i : i + config['sequence_length'], input_feature_indices].T
        input_tensor = torch.tensor(current_input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(config['device'])
        with torch.no_grad():
            predicted_toc_scaled = model(input_tensor).cpu().item()
        
        predicted_toc_denorm = output_feature_scaler.inverse_transform(np.array([[predicted_toc_scaled]])).flatten()[0]
        simulated_predictions.append(predicted_toc_denorm)

        if predicted_toc_denorm >= config['warning_threshold']:
            warning_status = "!!! WARNING !!!"
            simulated_warnings.append(1)
        else:
            warning_status = "Normal"
            simulated_warnings.append(0)
        
        print(f"模拟步 {i+1}: 预测TOC = {predicted_toc_denorm:.4f}, 状态: {warning_status}")

    plt.figure(figsize=(14, 7))
    plt.plot(simulated_predictions, label='Simulated Predicted TOC', color='red', linestyle='--')
    plt.axhline(y=config['warning_threshold'], color='green', linestyle=':', label=f'Warning Threshold ({config["warning_threshold"]})')
    simulated_warning_indices = np.where(np.array(simulated_warnings) == 1)[0]
    plt.scatter(simulated_warning_indices, np.array(simulated_predictions)[simulated_warning_indices],
                color='purple', marker='^', s=70, label='Predicted Warning Event')
    plt.title('Simulated Real-time TOC Prediction and Warning')
    plt.xlabel('Simulated Time Step')
    plt.ylabel('TOC Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'simulated_realtime_prediction.png'))
    plt.show()

def calculate_warning_metrics(all_true_values_denorm, all_predictions_denorm, warning_threshold):
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

    print("\n--- 预警系统性能评估 (基于预测值和真实值) ---")
    print(f"预警阈值: {warning_threshold}")
    print(f"TP (真实TOC超标且预测超标): {TP}")
    print(f"FP (真实TOC未超标但预测超标，误报): {FP}")
    print(f"FN (真实TOC超标但预测未超标，漏报): {FN}")
    print(f"TN (真实TOC未超标且预测未超标): {TN}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1_score}