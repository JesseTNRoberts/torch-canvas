import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # (e, 256)
        x = F.relu(self.fc2(x))  # (e, 256)
        x = F.relu(self.fc3(x))  # (e, 256)
        x = F.relu(self.fc4(x))  # (e, 256)
        x = F.relu(self.fc5(x))  # (e, 256)
        x = F.relu(self.fc6(x))  # (e, 256)
        x = F.relu(self.fc7(x))  # (e, 256)
        x = F.relu(self.fc8(x))  # (e, 256)
        x = F.relu(self.fc9(x))  # (e, 256)
        return self.fc10(x)

    def loss(self, out, y):
        """Sum of: mean squared error between out and y; Huber loss between out and y."""
        return (
            F.mse_loss(out, y)
            + 0.5 * F.huber_loss(out, y, delta=0.5)
        )

    @torch.no_grad()
    def predict(self, x):
        """Inference: forward() as is (the loss works on the raw output)."""
        return self(x)


if __name__ == "__main__":
    model = MLP()
    print(sum(p.numel() for p in model.parameters()), "parameters")
    x = torch.randn(8, 784)
    out = model(x)
    print(out.shape)  # expected torch.Size([8, 10])
    y = torch.randn(8, 10)
    print(model.loss(out, y).item(), "loss")
