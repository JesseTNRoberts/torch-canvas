import torch
import torch.nn as nn
import torch.nn.functional as F


class Cell(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.ih = nn.Linear(d, h)
        self.hh = nn.Linear(h, h)
        self.out = nn.Linear(h, d)

    def forward(self, x, state):
        z = torch.tanh(self.ih(x) + self.hh(state))
        return self.out(z), z


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell1 = Cell(64, 128)
        self.cell2 = Cell(64, 128)
        self.head = nn.Linear(64, 10)

    def forward(self, x, h0):
        y, h1 = self.cell1(x, h0)
        y2, h2 = self.cell2(y, h1)
        return self.head(y2), h2
