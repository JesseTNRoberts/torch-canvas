# title: LinearReLU
# description: A Linear layer followed by ReLU; repeat it to make a deep MLP.
import torch.nn as nn
import torch.nn.functional as F


class LinearReLU(nn.Module):
    def __init__(self, d=256, d_out=256):
        super().__init__()
        self.fc = nn.Linear(d, d_out)

    def forward(self, x):
        return F.relu(self.fc(x))
