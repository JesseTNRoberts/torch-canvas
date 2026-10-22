"""Qwen2.5-VL weights for the Torch Canvas model (examples/networks/10_qwen2_5_vl_3b.json).

hf_to_canvas(state_dict) turns a Hugging Face transformers Qwen2_5_VLForConditionalGeneration state dict into the
exported model's, e.g. for the released checkpoint:

    from transformers import Qwen2_5_VLForConditionalGeneration
    from model import Qwen2_5_VLForConditionalGeneration as Canvas
    hf = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="float32")
    model = Canvas()
    model.load_state_dict(hf_to_canvas(hf.state_dict()))

Run as a script it checks the claim that the two give the same output: it builds the reference model at a tiny size
with random weights, copies them over, and compares the logits on images of several sizes.
    python tools/qwen2_5_vl_weights.py path/to/tiny/model.py
"""
import re
import sys

import torch


def hf_to_canvas(sd, blocks_per_stage=8):
    out = {}
    for k, v in sd.items():
        k = k.replace("model.", "", 1) if k.startswith("model.") else k
        if k == "visual.patch_embed.proj.weight":  # Conv3d(3, d, (2, 14, 14), stride=kernel) == Linear(1176, d) on flat patches
            out["patch_embed.weight"] = v.reshape(v.shape[0], -1)
            continue
        m = re.match(r"visual\.blocks\.(\d+)\.(.*)", k)
        if m:
            i, rest = int(m.group(1)), m.group(2)
            s, j = divmod(i, blocks_per_stage)
            pre = f"stages.{s}.full." if j == blocks_per_stage - 1 else f"stages.{s}.blocks.{j}."
            if rest.startswith("attn.qkv."):  # one Linear(d, 3d) → q, k, v
                for name, part in zip("qkv", v.chunk(3, dim=0)):
                    out[pre + name + "." + rest.split(".")[-1]] = part
                continue
            rest = rest.replace("attn.proj.", "proj.").replace("mlp.", "")
            out[pre + rest] = v
            continue
        if k.startswith("visual.merger."):
            out["merger." + k[len("visual.merger."):].replace("mlp.0.", "fc1.").replace("mlp.2.", "fc2.")] = v
            continue
        if k.startswith("language_model."):
            k = k[len("language_model."):].replace("self_attn.", "").replace("mlp.", "")
            out[k] = v
            continue
        if k == "lm_head.weight":
            out[k] = v
            continue
        raise KeyError(f"unexpected key {k}")
    return out


def check(model_py):
    import importlib.util
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    spec = importlib.util.spec_from_file_location("canvas_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(
        vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True, rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
        image_token_id=999, video_token_id=998, vision_start_token_id=997, vision_end_token_id=996,
        vision_config=dict(depth=32, hidden_size=64, intermediate_size=96, num_heads=4, out_hidden_size=256, patch_size=14, temporal_patch_size=2,
                           spatial_merge_size=2, window_size=112, fullatt_block_indexes=[7, 15, 23, 31], in_channels=3),
    )
    hf = Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation="eager").eval()
    with torch.no_grad():  # random weights everywhere, biases and norms included
        for p in hf.parameters():
            p.normal_(0, 0.05)
        for n, p in hf.named_parameters():
            if "norm" in n or "ln_q" in n:
                p.add_(1.0)
    model = mod.Qwen2_5_VLForConditionalGeneration().eval()
    missing, unexpected = model.load_state_dict(hf_to_canvas(hf.state_dict()), strict=True), None

    # three images of different sizes (one of 2 frames) in two sequences; partial windows at the edges
    grid = torch.tensor([[1, 16, 20], [1, 12, 8], [2, 8, 8]])
    n_patch = int(grid.prod(-1).sum())
    pixels = torch.randn(n_patch, 3 * 2 * 14 * 14)
    tok = lambda *s: torch.randint(0, 990, s)
    img = lambda m: torch.full((m,), 999)
    row0 = torch.cat([tok(5), img(80), tok(7)])
    row1 = torch.cat([tok(3), img(24), tok(4), img(32), tok(29)])
    ids = torch.stack([row0, row1])
    with torch.no_grad():
        ref = hf(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(ids == 999).int()).logits
        ours = model(pixels, grid, ids)
    diff = (ref - ours).abs().max().item()
    print(f"logits {tuple(ours.shape)}, largest difference {diff:.2e} (logits around {ref.abs().mean().item():.2f})")
    print(f"same predictions: {bool((ref.argmax(-1) == ours.argmax(-1)).all())}")
    return diff


if __name__ == "__main__":
    d = check(sys.argv[1])
    sys.exit(0 if d < 1e-4 else 1)
