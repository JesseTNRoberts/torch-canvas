# Checks the VAE example against the model in PyTorch's examples/vae/main.py: copies its weights into the exported
# model, draws the same noise, and compares the reconstruction and the loss (examples/vae sums over the batch; the
# exported loss() averages over it).
# usage: python tools/check_vae.py model.py
import importlib.util, sys, torch, torch.nn as nn, torch.nn.functional as F
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class RefVAE(nn.Module):  # PyTorch examples/vae/main.py
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400); self.fc21 = nn.Linear(400, 20); self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400); self.fc4 = nn.Linear(400, 784)
    def encode(self, x): h1 = F.relu(self.fc1(x)); return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar): std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z): h3 = F.relu(self.fc3(z)); return torch.sigmoid(self.fc4(h3))
    def forward(self, x): mu, logvar = self.encode(x.view(-1, 784)); z = self.reparameterize(mu, logvar); return self.decode(z), mu, logvar

def ref_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

torch.manual_seed(0)
ref, ours = RefVAE(), m.VAE()
print('parameters', sum(p.numel() for p in ours.parameters()), 'examples/vae', sum(p.numel() for p in ref.parameters()))
ours.load_state_dict(ref.state_dict(), strict=True)
x = torch.rand(16, 1, 28, 28)
torch.manual_seed(1); r, mu, lv = ref(x)
torch.manual_seed(1); logits, mu2, lv2 = ours(x.view(-1, 784))
d = (torch.sigmoid(logits) - r).abs().max().item(); print('reconstruction: max |difference|', d); assert d < 1e-6
a, b = ref_loss(r, x, mu, lv).item() / 16, ours.loss(logits, mu2, lv2, x.view(-1, 784)).item()
print('loss per example', a, b); assert abs(a - b) < 1e-3 * abs(a)
print('same outputs and loss')
