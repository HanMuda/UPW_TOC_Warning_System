import torch
import torch.nn as nn
import torch.nn.functional as F

# --- TCN 基础模块 ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# --- Attention 模块 ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, rnn_outputs, final_hidden_state):
        final_hidden_state_expanded = final_hidden_state.unsqueeze(1).repeat(1, rnn_outputs.size(1), 1)
        combined = torch.cat((rnn_outputs, final_hidden_state_expanded), dim=2)
        attn_weights = self.v(torch.tanh(self.attn(combined)))
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * rnn_outputs, dim=1)
        return context_vector, attn_weights

# ==========================================================
# --- 模型库 (Model Zoo) ---
# ==========================================================

# 模型1: 单一LSTM模型
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(LSTM_Model, self).__init__()
        # LSTM需要 [Batch, SeqLen, Features]
        self.input_format = 'NLC' 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out.squeeze(1)

# 模型2: 单一TCN模型
class TCN_Model(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_Model, self).__init__()
        # TCN需要 [Batch, Features, SeqLen]
        self.input_format = 'NCL' 
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        tcn_out = self.tcn(x)
        # 使用最后一个时间步的输出进行预测
        last_time_step_out = tcn_out[:, :, -1]
        out = self.fc(last_time_step_out)
        return out.squeeze(1)

# 模型3: TCN-LSTM-Attention 复合模型
class TCN_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, tcn_num_channels, tcn_kernel_size, tcn_dropout,
                 lstm_hidden_size, lstm_num_layers, lstm_dropout):
        super(TCN_LSTM_Attention_Model, self).__init__()
        # 复合模型输入先给TCN，所以是 'NCL'
        self.input_format = 'NCL'
        self.tcn = TemporalConvNet(input_size, tcn_num_channels, tcn_kernel_size, tcn_dropout)
        tcn_output_channels = tcn_num_channels[-1]
        self.lstm = nn.LSTM(tcn_output_channels, lstm_hidden_size, lstm_num_layers,
                              batch_first=True, dropout=lstm_dropout if lstm_num_layers > 1 else 0)
        self.attention = Attention(lstm_hidden_size)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        tcn_output = self.tcn(x)
        lstm_input = tcn_output.permute(0, 2, 1)
        rnn_outputs, (hn, cn) = self.lstm(lstm_input)
        final_hidden_state = hn[-1, :, :]
        context_vector, _ = self.attention(rnn_outputs, final_hidden_state)
        out = self.fc(context_vector)
        return out.squeeze(1)

# --- 自定义损失函数 ---
class WeightedMSELoss(nn.Module):
    def __init__(self, warning_threshold: float, weight_factor: float):
        super(WeightedMSELoss, self).__init__()
        self.warning_threshold = warning_threshold
        self.weight_factor = weight_factor
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        weights = torch.ones_like(targets)
        high_toc_indices = targets >= self.warning_threshold
        weights[high_toc_indices] = self.weight_factor
        weighted_loss = mse_loss * weights
        return weighted_loss.mean()