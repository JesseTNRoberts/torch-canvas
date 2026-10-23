# title: SwiGLU
# description: Gated MLP without a residual: W_out (SiLU(x W) ⊙ x V). Output size can differ from input size.
# labels: hidden=hidden, d_out=output
# category: MLP blocks
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d=256, hidden=512, d_out=256):
        super().__init__()
        self.w = nn.Linear(d, hidden, bias=False)
        self.v = nn.Linear(d, hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_out, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w(x)) * self.v(x))
