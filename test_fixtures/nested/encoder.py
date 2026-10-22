import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.a = nn.Linear(d, h)
        self.b = nn.Linear(d, h)

    def forward(self, x):
        return torch.sigmoid(self.a(x)), self.b(x)


class Mix(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return self.proj(self.norm(x))


class Encoder(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.mix = Mix(d)
        self.gate = Gate(d, h)
        self.out = nn.Linear(h, d)

    def forward(self, x):
        m = self.mix(x)
        g, v = self.gate(m)
        return x + self.out(g * v)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(64, 256)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        return self.head(x)
