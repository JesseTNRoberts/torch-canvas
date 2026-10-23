# title: PreNormMLP
# description: Transformer-style MLP with a residual: x + W2 GELU(W1 LayerNorm(x)).
# labels: hidden=hidden
# category: MLP blocks
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
