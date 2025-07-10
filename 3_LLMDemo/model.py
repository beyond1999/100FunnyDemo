import torch
import torch.nn as nn
from torch.nn import functional as F
import math

d_model= 512
batch_size = 4
context_length = 16
num_heads = 8
head_size = d_model // num_heads
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_size)
        self.Wk = nn.Linear(d_model, head_size)
        self.Wv = nn.Linear(d_model, head_size)

    def forward(self, x):
        q = self.Wq(x) # q: [batch_size, context_length, head_size]
        k = self.Wk(x)
        v = self.Wv(x)

