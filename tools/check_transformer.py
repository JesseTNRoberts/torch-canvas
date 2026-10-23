# Checks the encoder-decoder Transformer example against torch.nn.Transformer (d_model 512, 8 heads, 6 + 6 layers,
# feed-forward 2048, post-norm) with a shared, scaled embedding, the same sinusoidal positions and a tied output
# projection: copies the weights (splitting nn.MultiheadAttention's packed q/k/v) and compares the logits (eval mode).
# usage: python tools/check_transformer.py model.py
import importlib.util, math, sys, torch, torch.nn as nn
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
V, D = 37000, 512

class Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.tr = nn.Transformer(D, 8, 6, 6, 2048, 0.1, batch_first=True)
    def forward(self, src, tgt):
        s = self.emb(src) * math.sqrt(D); s = s + m.sinusoidal_pe(s); t = self.emb(tgt) * math.sqrt(D); t = t + m.sinusoidal_pe(t)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        return self.tr(s, t, tgt_mask=mask, tgt_is_causal=True) @ self.emb.weight.T

torch.manual_seed(0)
ref, ours = Ref().eval(), m.Transformer().eval()
n_ref = sum(p.numel() for p in ref.parameters()); print('parameters', sum(p.numel() for p in ours.parameters()), 'nn.Transformer + embedding', n_ref)
R, sd = ref.state_dict(), {}
def attn(dst, src):
    W, B = R[src + 'in_proj_weight'].chunk(3), R[src + 'in_proj_bias'].chunk(3)
    for k, nm in enumerate('qkv'): sd[dst + nm + '.weight'], sd[dst + nm + '.bias'] = W[k], B[k]
    sd[dst + 'out.weight'], sd[dst + 'out.bias'] = R[src + 'out_proj.weight'], R[src + 'out_proj.bias']
def copy(dst, src):
    for t in ('weight', 'bias'): sd[dst + '.' + t] = R[src + '.' + t]
for i in range(6):
    p, q = f'tr.encoder.layers.{i}.', f'encoder.{i}.'
    attn(q + 'sa_', p + 'self_attn.')
    for nm in ('linear1', 'linear2', 'norm1', 'norm2'): copy(q + nm, p + nm)
    p, q = f'tr.decoder.layers.{i}.', f'decoder.{i}.'
    attn(q + 'sa_', p + 'self_attn.'); attn(q + 'ca_', p + 'multihead_attn.')
    for nm in ('linear1', 'linear2', 'norm1', 'norm2', 'norm3'): copy(q + nm, p + nm)
copy('encoder_norm', 'tr.encoder.norm'); copy('decoder_norm', 'tr.decoder.norm')
for k in ('embedding.weight', 'tgt_embedding.weight', 'generator.weight'): sd[k] = R['emb.weight']
ours.load_state_dict(sd, strict=True)
src, tgt = torch.randint(0, V, (2, 64)), torch.randint(0, V, (2, 48))
with torch.no_grad(): a, b = ref(src, tgt), ours(src, tgt)
d = (a - b).abs().max().item(); print('max |difference|', d, 'of', a.abs().max().item()); assert d <= 1e-4 * a.abs().max().item()
print('same outputs')
