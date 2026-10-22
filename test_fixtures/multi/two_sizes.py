import torch
import torch.nn as nn
import torch.nn.functional as F


class Split(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.a = nn.Linear(d, h)
        self.b = nn.Linear(d, h)

    def forward(self, x):
        return torch.relu(self.a(x)), torch.sigmoid(self.b(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = Split(64, 128)
        self.s2 = Split(128, 32)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        p, q = self.s1(x)
        r, t = self.s2(p * q)
        return self.head(r + t), q
