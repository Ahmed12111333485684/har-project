"""
Self-supervised pre-training of the Transformer by masked-patch reconstruction.
Uses ONLY unlabeled TRAIN-subject windows (no labels, no validation, no test subjects).
  python -m src.pretrain_ssl --epochs 40 --out checkpoints/transformer_ssl.pt
Then fine-tune with:  --grid pretrained=null,checkpoints/transformer_ssl.pt --grid label_fraction=0.1,1.0
"""
import argparse
import numpy as np, torch
from src.datasets.augment import build_augmenter
from src.datasets.har import HARDataset, get_splits
from src.models import build_model
from src.utils import ROOT, load_config, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "transformer.yaml"))
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--mask-ratio", type=float, default=0.4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="checkpoints/transformer_ssl.pt")
a = ap.parse_args()

cfg = load_config(a.config)
set_seed(a.seed)
d = cfg["data"]
sp = get_splits(d["root"], d["channels"], d["window_len"], d["val_subjects"], d["split_seed"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("transformer", sp.X_train.shape[-1], 6, **cfg["model"]["params"]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
loader = torch.utils.data.DataLoader(HARDataset(sp.X_train, sp.y_train, build_augmenter(cfg["augment"]), a.seed),
                                     batch_size=128, shuffle=True, drop_last=True)
for ep in range(1, a.epochs + 1):
    model.train(); tot = n = 0
    for xb, _ in loader:                       # labels are ignored
        xb = xb.to(dev)
        opt.zero_grad(); loss = model.ssl_loss(xb, a.mask_ratio); loss.backward(); opt.step()
        tot += loss.item(); n += 1
    if ep % 5 == 0 or ep == 1:
        print(f"ssl epoch {ep:3d}  recon MSE {tot/n:.4f}", flush=True)
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(model.state_dict(), a.out)
print("saved", a.out)
