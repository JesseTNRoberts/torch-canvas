# Checks the ViT-B/16 example against torchvision's vit_b_16: copies torchvision's weights into the exported model
# and compares outputs.  usage: python tools/check_vit.py vit.py
import importlib.util, sys, torch, torchvision
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
ref = torchvision.models.vit_b_16().eval(); ours = m.ViT_B_16().eval()
print('parameters', sum(p.numel() for p in ours.parameters()), 'torchvision', sum(p.numel() for p in ref.parameters()))
# torchvision starts the head at zero, which would hide any difference
with torch.no_grad(): torch.nn.init.normal_(ref.heads.head.weight, std=0.02); torch.nn.init.normal_(ref.heads.head.bias, std=0.02); torch.nn.init.normal_(ref.class_token, std=0.5)
R = ref.state_dict(); sd = {}
sd['conv_proj.weight'], sd['conv_proj.bias'] = R['conv_proj.weight'], R['conv_proj.bias']
sd['class_token'] = R['class_token']; sd['pos_embedding.weight'] = R['encoder.pos_embedding'][0]
sd['ln.weight'], sd['ln.bias'] = R['encoder.ln.weight'], R['encoder.ln.bias']
sd['head.weight'], sd['head.bias'] = R['heads.head.weight'], R['heads.head.bias']
for i in range(12):
    p, q = f'encoder.layers.encoder_layer_{i}.', f'layers.{i}.'
    for a, b in (('ln_1', 'ln_1'), ('ln_2', 'ln_2'), ('mlp.0', 'fc1'), ('mlp.3', 'fc2'), ('self_attention.out_proj', 'out_proj')):
        for t in ('weight', 'bias'): sd[q + b + '.' + t] = R[p + a + '.' + t]
    W, B = R[p + 'self_attention.in_proj_weight'].chunk(3), R[p + 'self_attention.in_proj_bias'].chunk(3)
    for k, nm in enumerate(('q_proj', 'k_proj', 'v_proj')): sd[q + nm + '.weight'], sd[q + nm + '.bias'] = W[k], B[k]
missing = ours.load_state_dict(sd, strict=True); x = torch.randn(2, 3, 224, 224)
with torch.no_grad(): a, b = ref(x), ours(x)
print('max |difference|', (a - b).abs().max().item(), 'of', a.abs().max().item()); assert torch.allclose(a, b, atol=1e-4)
print('same outputs')
