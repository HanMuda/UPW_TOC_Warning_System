import torch
import numpy as np
import random

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

def set_seeds(seed: int = 42):
    """
    为所有相关的库设置随机种子。
    """
    # Python 内置的随机库
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # 如果使用GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 适用于多GPU情况
        
        # 将cuDNN设置为确定性模式
        # 这可能会稍微降低训练速度，但对于复现性是必需的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"所有随机种子已设置为: {seed}")