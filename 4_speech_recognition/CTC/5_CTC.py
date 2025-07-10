import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(CTCModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional -> *2

    def forward(self, x):
        # x: [batch, time, feature]
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)  # CTC expects log probabilities
        return x

# 参数定义
input_dim = 40          # 特征维度（如梅尔频率）
hidden_dim = 128        # LSTM隐层维度
output_dim = 29         # 28个字符 + 1个blank

model = CTCModel(input_dim, hidden_dim, output_dim)

# 假设输入：batch=2，时间步=100，特征维度=40
inputs = torch.randn(2, 100, input_dim)  # [B, T, D]
outputs = model(inputs)  # 输出：[B, T, C]
print(outputs.shape)  # torch.Size([2, 100, 29])
