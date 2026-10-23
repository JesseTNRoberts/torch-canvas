# Checks the image examples against independent references:
#   ResNet-50 (examples/networks/12_resnet50.json) against torchvision.models.resnet50
#   YOLOv3 (examples/networks/13_yolov3.json) against a model built straight from Darknet's yolov3.cfg
# Both get the reference's weights copied in order (with randomized BatchNorm statistics) and must give the same outputs.
# needs: node, torch, torchvision; yolov3.cfg is downloaded from github.com/pjreddie/darknet the first time.
# usage: python tools/check_image_models.py
import importlib.util, os, subprocess, sys, tempfile, urllib.request, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TMP = tempfile.mkdtemp()
def generate(example, out):
    js = "const NN=require(process.argv[1]),fs=require('fs');fs.writeFileSync(process.argv[3],NN.generate(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).graph).code)"
    subprocess.run(['node', '-e', js, os.path.join(ROOT, 'core.js'), os.path.join(ROOT, 'examples', 'networks', example), out], check=True)
    return out
CFG = os.path.join(HERE, 'yolov3.cfg')
if not os.path.exists(CFG): urllib.request.urlretrieve('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', CFG)
torch.manual_seed(0)
def load(path, cls):
    spec = importlib.util.spec_from_file_location(cls, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return getattr(m, cls)
def copy_in_order(src, dst):
    a, b = list(src.state_dict().items()), list(dst.state_dict().items())
    assert len(a) == len(b), (len(a), len(b))
    sd = {}
    for (ka, va), (kb, vb) in zip(a, b):
        assert va.shape == vb.shape, (ka, kb, va.shape, vb.shape)
        sd[kb] = va.clone()
    dst.load_state_dict(sd)
    return len(a)
def randomize_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5); mod.running_var.uniform_(0.5, 1.5); mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.2, 0.2)

# ResNet-50
import torchvision
ref = torchvision.models.resnet50(weights=None); randomize_bn(ref); ref.eval()
ours = load(generate('12_resnet50.json', os.path.join(TMP, 'resnet50.py')), 'ResNet50')(); ours.eval()
n = copy_in_order(ref, ours)
p_ref, p_ours = sum(p.numel() for p in ref.parameters()), sum(p.numel() for p in ours.parameters())
x = torch.randn(2, 3, 224, 224)
with torch.no_grad(): d = (ref(x) - ours(x)).abs().max().item()
print(f"ResNet-50: {p_ours:,} parameters (torchvision {p_ref:,}), {n} tensors copied in order, max output difference {d:.2e}")
assert p_ref == p_ours and d < 1e-4

# YOLOv3 from yolov3.cfg
def parse_cfg(path):
    blocks, cur = [], None
    for line in open(path):
        line = line.split('#')[0].strip()
        if not line: continue
        if line.startswith('['): cur = {'type': line[1:-1]}; blocks.append(cur)
        else: k, v = line.split('='); cur[k.strip()] = v.strip()
    return blocks
class Darknet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = parse_cfg(cfg)[1:]
        self.mods = nn.ModuleList(); ch = [3]; c = 3
        for b in self.blocks:
            m = nn.Sequential()
            if b['type'] == 'convolutional':
                bn_ = int(b.get('batch_normalize', 0)); f = int(b['filters']); k = int(b['size']); s = int(b['stride']); p = k // 2 if int(b['pad']) else 0
                m.add_module('conv', nn.Conv2d(c, f, k, s, p, bias=not bn_))
                if bn_: m.add_module('bn', nn.BatchNorm2d(f))
                if b['activation'] == 'leaky': m.add_module('act', nn.LeakyReLU(0.1))
                c = f
            elif b['type'] == 'upsample': m.add_module('up', nn.Upsample(scale_factor=int(b['stride']), mode='nearest'))
            elif b['type'] == 'route':
                ls = [int(v) for v in b['layers'].split(',')]; ls = [l if l >= 0 else len(ch) - 1 + l for l in ls]; b['_ls'] = ls; c = sum(ch[l + 1] for l in ls)
            elif b['type'] == 'shortcut': b['_from'] = int(b['from'])
            self.mods.append(m); ch.append(c)
    def forward(self, x):
        outs, dets = [], []
        for i, (b, m) in enumerate(zip(self.blocks, self.mods)):
            t = b['type']
            if t in ('convolutional', 'upsample'): x = m(x)
            elif t == 'route': x = torch.cat([outs[l] for l in b['_ls']], 1)
            elif t == 'shortcut': x = x + outs[i + b['_from']]
            elif t == 'yolo': dets.append(x)
            outs.append(x)
        return dets
ref = Darknet(CFG); randomize_bn(ref); ref.eval()
ours = load(generate('13_yolov3.json', os.path.join(TMP, 'yolov3.py')), 'YOLOv3')(); ours.eval()
n = copy_in_order(ref, ours)
p_ref, p_ours = sum(p.numel() for p in ref.parameters()), sum(p.numel() for p in ours.parameters())
x = torch.randn(1, 3, 416, 416)
with torch.no_grad(): a, b = ref(x), ours(x)
d = max((u - v).abs().max().item() for u, v in zip(a, b))
print(f"YOLOv3: {p_ours:,} parameters (yolov3.cfg {p_ref:,}), {n} tensors copied in order, outputs {[tuple(o.shape) for o in b]}, max difference {d:.2e}")
assert p_ref == p_ours and d < 1e-4 and len(a) == 3

# basic CNN
M = load(generate('11_basic_cnn.json', os.path.join(TMP, 'cnn.py')), 'MNISTConvNet')(); x = torch.randn(4, 1, 28, 28); y = torch.randint(0, 10, (4,))
out = M(x); print(f"Basic CNN: {sum(p.numel() for p in M.parameters()):,} parameters, logits {tuple(out.shape)}, loss {M.loss(out, y).item():.3f}, predict sums to {M.predict(x).sum(-1)[0].item():.3f}")
