# Checks the U-Net example against milesial/Pytorch-UNet (UNet(3, 2, bilinear=False)): copies its weights into the
# exported model, layer by layer in order, and compares the outputs (BatchNorm in training mode and in eval mode).
# usage: python tools/check_unet.py model.py path/to/Pytorch-UNet
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, sys.argv[2]); from unet import UNet
torch.manual_seed(0)
ref, ours = UNet(3, 2, bilinear=False), m.UNet()
print('parameters', sum(p.numel() for p in ours.parameters()), 'Pytorch-UNet', sum(p.numel() for p in ref.parameters()))
R, O = ref.state_dict(), ours.state_dict()
assert len(R) == len(O) and all(a.shape == b.shape for a, b in zip(R.values(), O.values())), 'layers differ'
ours.load_state_dict(dict(zip(O.keys(), R.values())))
x = torch.randn(2, 3, 256, 256)
for mode in ('train', 'eval'):
    getattr(ref, mode)(); getattr(ours, mode)()
    with torch.no_grad(): a, b = ref(x), ours(x)
    d = (a - b).abs().max().item(); print(mode, 'max |difference|', d, 'of', a.abs().max().item()); assert d <= 1e-4 * a.abs().max().item()
print('same outputs')
