import torch
import torch.nn as nn
import torch.optim as optim

# 模拟一组输入数据
batch_size = 4
seq_len = 5
input_size = 10
hidden_size = 16
num_classes = 2

X = torch.randn(batch_size, seq_len, input_size)
y = torch.randint(0, num_classes, (batch_size,))


# 定义模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn.squeeze(0))
        return out


model = SimpleLSTM(input_size, hidden_size, num_classes)

# 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练一轮
model.train()
optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()

print("训练前 loss：", loss.item())
print("LSTM 输入权重梯度前 5 项：", model.lstm.weight_ih_l0.grad[:5])

optimizer.step()
