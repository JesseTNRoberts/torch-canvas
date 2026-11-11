import marimo

app = marimo.App()

with app.setup:
    import torch
    import torch.nn as nn


@app.class_definition
class Net(nn.Module):
    """Tiny net."""

    def __init__(self, n_in=8, n_out=4):
        super().__init__()
        # keep me
        self.fc1 = nn.Linear(n_in, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, n_out)

    def forward(self, x):
        """Doc."""
        return self.fc2(self.act(self.fc1(x)))

    def predict(self, x):
        return self.forward(x).argmax(1)


if __name__ == "__main__":
    app.run()
