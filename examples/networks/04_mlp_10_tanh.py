# title: MLP, 10 layers, Tanh
# description: The same 10-layer MLP with Tanh between the layers.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in=784, width=256, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, width)
        self.fc7 = nn.Linear(width, width)
        self.fc8 = nn.Linear(width, width)
        self.fc9 = nn.Linear(width, width)
        self.fc10 = nn.Linear(width, n_classes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        x = torch.tanh(self.fc9(x))
        return self.fc10(x)
