# title: MLP, 10 layers, ReLU
# description: A plain deep MLP: 10 Linear layers with ReLU between them. Good for seeing how depth and init affect the signal.
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        return self.fc10(x)
