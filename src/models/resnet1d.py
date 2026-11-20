"""
ADVANCED family: ResNet-1D with Squeeze-and-Excitation blocks and learned
temporal attention pooling. `use_se` / `use_attn_pool` allow ablations.
"""
import torch
import torch.nn as nn


class SE1d(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):                                  # (B, C, T)
        return x * self.fc(x.mean(2)).unsqueeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, kernel, stride, use_se):
        super().__init__()
        p = kernel // 2
        self.c1 = nn.Conv1d(cin, cout, kernel, stride, p, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, kernel, 1, p, bias=False);     self.b2 = nn.BatchNorm1d(cout)
        self.se = SE1d(cout) if use_se else nn.Identity()
        self.short = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                      if (cin != cout or stride != 1) else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.se(self.b2(self.c2(h)))
        return self.act(h + self.short(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels, n_classes, widths=(32, 64, 128), blocks=(1, 1, 1),
                 kernel=5, dropout=0.3, use_se=True, use_attn_pool=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, widths[0], 7, padding=3, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU())
        layers, c = [], widths[0]
        for i, (w, nb) in enumerate(zip(widths, blocks)):
            for j in range(nb):
                layers.append(ResBlock1D(c, w, kernel, 2 if (j == 0 and i > 0) else 1, use_se))
                c = w
        self.body = nn.Sequential(*layers)
        self.use_attn_pool = use_attn_pool
        self.attn = nn.Linear(c, 1) if use_attn_pool else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, n_classes)
        self.last_pool_weights = None

    def forward(self, x):                                   # (B, T, C)
        h = self.body(self.stem(x.transpose(1, 2))).transpose(1, 2)   # (B, T', c)
        if self.use_attn_pool:
            w = self.attn(h).softmax(dim=1)                  # (B, T', 1)
            self.last_pool_weights = w.detach()
            f = (w * h).sum(1)
        else:
            f = h.mean(1)
        return self.fc(self.drop(f))
