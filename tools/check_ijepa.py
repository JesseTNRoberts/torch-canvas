# Checks the small I-JEPA example against facebookresearch/ijepa (src/models/vision_transformer.py): a ViT-Tiny-style
# encoder (32 × 32 images, 4 × 4 patches, width 192, 6 layers, 3 heads) and predictor (width 96, 3 layers), copies
# the weights in (splitting each block's packed qkv) and compares the predictions and the targets for the same masks.
# Then one EMA step: ema_update(progress) against I-JEPA's momentum update of its target encoder.
# usage: python tools/check_ijepa.py model.py path/to/ijepa
import copy, importlib.util, sys, torch, torch.nn.functional as F
from functools import partial
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, sys.argv[2])
from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from src.masks.utils import apply_masks

torch.manual_seed(0)
LN = partial(torch.nn.LayerNorm, eps=1e-6)
enc = VisionTransformer(img_size=[32], patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=LN)
pred = VisionTransformerPredictor(num_patches=64, embed_dim=192, predictor_embed_dim=96, depth=3, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=LN)
tgt = copy.deepcopy(enc)
ours = m.IJEPA()
n_ref = sum(p.numel() for mod in (enc, pred, tgt) for p in mod.parameters())
print('parameters', sum(p.numel() for p in ours.parameters()), 'ijepa (encoder + predictor + target encoder)', n_ref)

def blocks(dst, src, R, n):
    for i in range(n):
        p, q = f'{src}.{i}.', f'{dst}.{i}.'
        W, B = R[p + 'attn.qkv.weight'].chunk(3), R[p + 'attn.qkv.bias'].chunk(3)
        for k, nm in enumerate(('q_proj', 'k_proj', 'v_proj')): sd[q + nm + '.weight'], sd[q + nm + '.bias'] = W[k], B[k]
        for a, b in (('norm1', 'norm1'), ('attn.proj', 'proj'), ('norm2', 'norm2'), ('mlp.fc1', 'fc1'), ('mlp.fc2', 'fc2')):
            for t in ('weight', 'bias'): sd[q + b + '.' + t] = R[p + a + '.' + t]
def encoder(pre, patch, R):
    sd[patch + '.weight'], sd[patch + '.bias'] = R['patch_embed.proj.weight'], R['patch_embed.proj.bias']
    blocks(pre + '.blocks', 'blocks', R, 6)
    sd[pre + '.norm.weight'], sd[pre + '.norm.bias'] = R['norm.weight'], R['norm.bias']
sd = {}
E, P, T = enc.state_dict(), pred.state_dict(), tgt.state_dict()
encoder('encoder', 'patch_embed', E); encoder('target_encoder', 'target_patch_embed', T)
sd['pos_embed'] = E['pos_embed']
for k in ('predictor_embed.weight', 'predictor_embed.bias', 'mask_token', 'predictor_pos_embed', 'predictor_norm.weight', 'predictor_norm.bias', 'predictor_proj.weight', 'predictor_proj.bias'): sd[k] = P[k]
blocks('predictor_blocks', 'predictor_blocks', P, 3)
ours.load_state_dict(sd, strict=True)
for a, b in (('pos_embed', E['pos_embed']), ('predictor_pos_embed', P['predictor_pos_embed'])):
    d = (getattr(m.IJEPA(), a) - b).abs().max().item(); print(a, 'as written by sincos_2d: max |difference| from I-JEPA\'s', d); assert d < 1e-6

imgs = torch.randn(4, 3, 32, 32)
perm = torch.rand(4, 64).argsort(dim=1); ctx, tg = perm[:, :40].sort(1).values, perm[:, 40:56].sort(1).values
with torch.no_grad():
    h = apply_masks(F.layer_norm(tgt(imgs), (192,)), [tg])
    z = pred(enc(imgs, [ctx]), [ctx], [tg])
    zp, hp = ours(imgs, ctx, tg)
for nm, a, b in (('predictions', z, zp), ('targets', h, hp)):
    d = (a - b).abs().max().item(); print(nm, 'max |difference|', d, 'of', a.abs().max().item()); assert d <= 1e-5 * a.abs().max().item()
# one optimizer step on the context encoder, then the EMA update, both ways
with torch.no_grad():
    for mod in (enc, ours.encoder, ours.patch_embed):
        for p in mod.parameters(): p.add_(0.01)
    mom = 0.996 + 0.3 * (1.0 - 0.996)
    for q, k in zip(enc.parameters(), tgt.parameters()):
        k.mul_(mom).add_((1. - mom) * q.detach())
    ours.ema_update(0.3)
    h2 = apply_masks(F.layer_norm(tgt(imgs), (192,)), [tg]); _, hp2 = ours(imgs, ctx, tg)
d = (h2 - hp2).abs().max().item(); print('targets after an EMA step: max |difference|', d); assert d <= 1e-5 * h2.abs().max().item()
print('same outputs, and the same EMA update')
