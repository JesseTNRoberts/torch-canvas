import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, base=10000.0):
    """Rotary position embedding on (B, n_head, T, D): rotates feature pairs (i, i + D/2) by position-dependent angles."""
    t, d = x.size(-2), x.size(-1)
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv_freq)
    ang = torch.cat((ang, ang), dim=-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * ang.cos().to(x.dtype) + torch.cat((-x2, x1), dim=-1) * ang.sin().to(x.dtype)


def causal_mask(scores):
    """Hide future positions: scores above the diagonal become -inf."""
    t = scores.size(-1)
    future = torch.ones(t, t, dtype=torch.bool, device=scores.device).triu(1)
    return scores.masked_fill(future, float("-inf"))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, d=64, n_head=4, ff=172):
        super().__init__()
        self.n_head = n_head
        self.input_layernorm = nn.RMSNorm(d, eps=1e-5)
        self.q_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        self.k_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        self.v_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.o_proj.weight, std=0.02)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=1e-5)
        self.gate_proj = nn.Linear(d, ff, bias=False)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        self.up_proj = nn.Linear(d, ff, bias=False)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        self.down_proj = nn.Linear(ff, d, bias=False)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, x):
        n = self.input_layernorm(x)  # (e, t, d)
        q = self.q_proj(n).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        q = apply_rope(q)  # (e, n_head, t, d/n_head)
        q = q * q.size(-1) ** -0.5  # (e, n_head, t, d/n_head)
        k = self.k_proj(n).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        k = apply_rope(k)  # (e, n_head, t, d/n_head)
        att = q @ k.transpose(-2, -1)  # (e, n_head, t, t)
        att = causal_mask(att)  # (e, n_head, t, t)
        att = torch.softmax(att, dim=-1)  # (e, n_head, t, t)
        v = self.v_proj(n).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        y = att @ v  # (e, n_head, t, d/n_head)
        y = y.transpose(1, 2).flatten(2)  # (e, t, d)
        x = x + self.o_proj(y)  # (e, t, d)
        n = self.post_attention_layernorm(x)  # (e, t, d)
        h = F.silu(self.gate_proj(n)) * self.up_proj(n)  # (e, t, ff)
        return x + self.down_proj(h)


class LlamaForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 64)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList([LlamaDecoderLayer(64, 4, 172) for _ in range(2)])
        self.norm = nn.RMSNorm(64, eps=1e-5)
        self.lm_head = nn.Linear(64, 100, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)  # (e, 64, 64)
        for blk in self.layers:  # (e, 64, 64)
            x = blk(x)
        n = self.norm(x)  # (e, 64, 64)
        return self.lm_head(n)

    def loss(self, logits, input_ids):
        """Next-token cross-entropy: position t of logits is scored on token t + 1 of input_ids."""
        return F.cross_entropy(logits[:, :-1].flatten(0, -2), input_ids[:, 1:].flatten())

    @torch.no_grad()
    def predict(self, input_ids):
        """Inference: forward() followed by softmax, which the loss leaves out."""
        return torch.softmax(self(input_ids), dim=-1)


if __name__ == "__main__":
    model = LlamaForCausalLM()
    print(sum(p.numel() for p in model.parameters()), "parameters")
    input_ids = torch.randint(0, 100, (8, 64))
    logits = model(input_ids)
    print(logits.shape)  # expected torch.Size([8, 64, 100])
    print(model.loss(logits, input_ids).item(), "loss")
