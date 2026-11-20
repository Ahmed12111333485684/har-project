"""
MODERN family: Transformer encoder for sensor windows.

The multi-head self-attention is WRITTEN FROM SCRATCH (no nn.MultiheadAttention)
- this is our "own key component" (requirement 5).

  window (T, C) --patch Conv1d--> tokens --(+sinusoidal pos)--> [CLS]+tokens
        --> N x pre-LN encoder blocks --> CLS --> linear head

Also supports self-supervised pre-training by masked patch reconstruction
(`ssl_loss`), so we can answer "pretrained vs scratch".
`mixer='none'` removes attention (token mixing) for the ablation study.
"""
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.keep_attn, self.last_attn = False, None

    def forward(self, x):                                         # (B, N, D)
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)      # (B, h, N, N)
        attn = scores.softmax(dim=-1)
        if self.keep_attn:                                         # for the GUI
            self.last_attn = attn.detach()
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult, dropout, mixer="attention"):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mixer = MultiHeadSelfAttention(d_model, n_heads, dropout) if mixer == "attention" else None
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.mixer is not None:
            x = x + self.drop(self.mixer(self.ln1(x)))
        return x + self.drop(self.ff(self.ln2(x)))


def sinusoidal(n, d, device):
    pos = torch.arange(n, device=device).float()[:, None]
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, device=device).float() / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class HARTransformer(nn.Module):
    def __init__(self, in_channels, n_classes, d_model=64, n_heads=4, n_layers=3,
                 ff_mult=2, patch=8, dropout=0.1, mixer="attention"):
        super().__init__()
        self.patch, self.in_channels, self.d_model = patch, in_channels, d_model
        self.use_cls = mixer == "attention"   # without attention the CLS token could never see the signal
        self.embed = nn.Conv1d(in_channels, d_model, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ff_mult, dropout, mixer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.recon_head = nn.Linear(d_model, patch * in_channels)   # SSL only

    def tokens(self, x, mask=None):                                 # x: (B, T, C)
        z = self.embed(x.transpose(1, 2)).transpose(1, 2)           # (B, N, D)
        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(z), z)
        z = z + sinusoidal(z.size(1), self.d_model, z.device)
        z = torch.cat([self.cls.expand(z.size(0), -1, -1), z], dim=1)
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)

    def forward(self, x):
        z = self.tokens(x)
        # attention model: classify from CLS; no-attention ablation: mean-pool the patch tokens
        return self.head(z[:, 0] if self.use_cls else z[:, 1:].mean(1))

    def ssl_loss(self, x, mask_ratio=0.4):
        """Masked-patch reconstruction (MSE only on masked patches)."""
        B, T, C = x.shape
        N = T // self.patch
        x = x[:, :N * self.patch]
        mask = torch.rand(B, N, device=x.device) < mask_ratio
        mask[:, 0] |= ~mask.any(dim=1)                              # at least one masked patch
        pred = self.recon_head(self.tokens(x, mask)[:, 1:])
        target = x.reshape(B, N, self.patch * C)
        err = ((pred - target) ** 2).mean(-1)
        return (err * mask).sum() / mask.sum()
