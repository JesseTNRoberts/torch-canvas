# title: Pre-norm residual MLP
# description: A transformer-style stack: four pre-LayerNorm GELU MLP blocks with residual connections, then a classifier.
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormMLP(nn.Module):
    def __init__(self, d=256, hidden=1024):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc_in = nn.Linear(d, hidden)
        self.fc_out = nn.Linear(hidden, d)

    def forward(self, x):
        return x + self.fc_out(F.gelu(self.fc_in(self.norm(x))))


class PreNormResidualMLP(nn.Module):
    def __init__(self, d_in=784, d=256, n_classes=10):
        super().__init__()
        self.embed = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList([PreNormMLP(d, 4 * d) for _ in range(4)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
