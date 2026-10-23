import torch
import torch.nn as nn
import torch.nn.functional as F


class BertLayer(nn.Module):
    def __init__(self, d=768, n_head=12, ff=3072):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(d, d)
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.zeros_(self.query.bias)
        self.key = nn.Linear(d, d)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.zeros_(self.key.bias)
        self.attn_drop = nn.Dropout(0.1)
        self.value = nn.Linear(d, d)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.zeros_(self.value.bias)
        self.attn_out = nn.Linear(d, d)
        nn.init.normal_(self.attn_out.weight, std=0.02)
        nn.init.zeros_(self.attn_out.bias)
        self.attn_out_drop = nn.Dropout(0.1)
        self.attn_norm = nn.LayerNorm(d, eps=1e-12)
        self.ff_in = nn.Linear(d, ff)
        nn.init.normal_(self.ff_in.weight, std=0.02)
        nn.init.zeros_(self.ff_in.bias)
        self.ff_out = nn.Linear(ff, d)
        nn.init.normal_(self.ff_out.weight, std=0.02)
        nn.init.zeros_(self.ff_out.bias)
        self.ff_drop = nn.Dropout(0.1)
        self.ff_norm = nn.LayerNorm(d, eps=1e-12)

    def forward(self, x):
        q = self.query(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        q = q * q.size(-1) ** -0.5  # (e, n_head, t, d/n_head)
        k = self.key(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        att = q @ k.transpose(-2, -1)  # (e, n_head, t, t)
        att = torch.softmax(att, dim=-1)  # (e, n_head, t, t)
        att = self.attn_drop(att)  # (e, n_head, t, t)
        v = self.value(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        y = att @ v  # (e, n_head, t, d/n_head)
        y = y.transpose(1, 2).flatten(2)  # (e, t, d)
        h = self.attn_out(y)  # (e, t, d)
        x = x + self.attn_out_drop(h)  # (e, t, d)
        x = self.attn_norm(x)  # (e, t, d)
        a = F.gelu(self.ff_in(x))  # (e, t, ff)
        h = self.ff_out(a)  # (e, t, d)
        x = x + self.ff_drop(h)  # (e, t, d)
        return self.ff_norm(x)


class BertForMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_emb = nn.Embedding(30522, 768)
        nn.init.normal_(self.word_emb.weight, std=0.02)
        self.pos_emb = nn.Embedding(512, 768)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.type_emb = nn.Embedding(2, 768)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.emb_norm = nn.LayerNorm(768, eps=1e-12)
        self.emb_drop = nn.Dropout(0.1)
        self.layer = nn.ModuleList([BertLayer(768, 12, 3072) for _ in range(12)])
        self.mlm_dense = nn.Linear(768, 768)
        nn.init.normal_(self.mlm_dense.weight, std=0.02)
        nn.init.zeros_(self.mlm_dense.bias)
        self.mlm_norm = nn.LayerNorm(768, eps=1e-12)
        self.decoder = nn.Linear(768, 30522)
        nn.init.zeros_(self.decoder.bias)
        self.decoder.weight = self.word_emb.weight

    def forward(self, input_ids, token_type_ids):
        x = self.pos_emb(torch.arange(input_ids.size(1), device=input_ids.device))  # (64, 768)
        x = self.word_emb(input_ids) + x  # (e, 64, 768)
        x = x + self.type_emb(token_type_ids)  # (e, 64, 768)
        x = self.emb_norm(x)  # (e, 64, 768)
        x = self.emb_drop(x)  # (e, 64, 768)
        for blk in self.layer:  # (e, 64, 768)
            x = blk(x)
        a = F.gelu(self.mlm_dense(x))  # (e, 64, 768)
        n = self.mlm_norm(a)  # (e, 64, 768)
        return self.decoder(n)

    def loss(self, logits, labels):
        """Masked-token cross-entropy: labels come from input_ids, masking 15% of them with id 103 (-100 marks the rest)."""
        return F.cross_entropy(logits.flatten(0, -2), labels.flatten(), ignore_index=-100)

    @torch.no_grad()
    def predict(self, input_ids, token_type_ids):
        """Inference: forward() followed by softmax, which the loss leaves out."""
        return torch.softmax(self(input_ids, token_type_ids), dim=-1)


if __name__ == "__main__":
    model = BertForMaskedLM()
    print(sum(p.numel() for p in model.parameters()), "parameters")
    input_ids = torch.randint(0, 30522, (8, 64))
    token_type_ids = torch.randint(0, 2, (8, 64))
    logits = model(input_ids, token_type_ids)
    print(logits.shape)  # expected torch.Size([8, 64, 30522])
    labels = torch.randint(0, 30522, (8, 64))
    print(model.loss(logits, labels).item(), "loss")
