import torch
import torch.nn as nn
# 输入维度: [batch_size, seq_len, input_dim]
input = torch.randn(32, 10, 128)  # 32 个样本，每个样本是长度为10的序列，每步特征是128维

rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

output, hn = rnn(input)

print("output shape:", output.shape)  # torch.Size([32, 10, 64])
print("hn shape:", hn.shape)          # torch.Size([1, 32, 64])