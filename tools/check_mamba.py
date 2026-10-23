# Checks the Mamba-130M example against transformers' MambaForCausalLM (state-spaces/mamba-130m's configuration,
# random weights): copies its weights into the exported model and compares the logits.
# usage: python tools/check_mamba.py model.py
import importlib.util, sys, torch
from transformers import MambaConfig, MambaForCausalLM
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
cfg = MambaConfig(vocab_size=50280, hidden_size=768, state_size=16, num_hidden_layers=24, expand=2, conv_kernel=4, layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, tie_word_embeddings=True)
ref = MambaForCausalLM(cfg).eval(); ours = m.Mamba().eval()
print('parameters', sum(p.numel() for p in ours.parameters()), 'transformers', sum(p.numel() for p in ref.parameters()))
R = ref.state_dict(); sd = {'embeddings.weight': R['backbone.embeddings.weight'], 'norm_f.weight': R['backbone.norm_f.weight'], 'lm_head.weight': R['backbone.embeddings.weight']}
di, N, r = 1536, 16, cfg.time_step_rank
for i in range(24):
    p, q = f'backbone.layers.{i}.', f'layers.{i}.'
    sd[q + 'norm.weight'] = R[p + 'norm.weight']
    sd[q + 'in_x.weight'], sd[q + 'in_z.weight'] = R[p + 'mixer.in_proj.weight'].chunk(2)
    sd[q + 'x_dt.weight'], sd[q + 'x_B.weight'], sd[q + 'x_C.weight'] = R[p + 'mixer.x_proj.weight'].split([r, N, N])
    for a in ('conv1d.weight', 'conv1d.bias', 'dt_proj.weight', 'dt_proj.bias', 'A_log', 'D', 'out_proj.weight'): sd[q + a] = R[p + 'mixer.' + a]
ours.load_state_dict(sd, strict=True)
ids = torch.randint(0, 50280, (2, 32))
# one block on its own: the same up to float32 rounding
x = torch.randn(2, 32, 768)
with torch.no_grad(): a, b = ref.backbone.layers[0](x), ours.layers[0](x)
a = a[0] if isinstance(a, tuple) else a
print('block 0: max |difference|', (a - b).abs().max().item(), 'of', a.abs().max().item()); assert (a - b).abs().max() <= 1e-6 * a.abs().max()
# the whole model: 24 blocks of float32 rounding (transformers runs the scan in its own order)
with torch.no_grad(): a, b = ref(ids).logits, ours(ids)
print('logits: max |difference|', (a - b).abs().max().item(), 'of', a.abs().max().item()); assert (a - b).abs().max() <= 1e-3 * a.abs().max()
print('argmax agrees on', (a.argmax(-1) == b.argmax(-1)).float().mean().item() * 100, '% of positions')
print('same outputs, up to float32 rounding')
