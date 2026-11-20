"""
Shared evaluation for ALL models: metrics, subject-level bootstrap CI,
cost measurements (params, size, inference latency).
CLI:  python -m src.evaluate --run-dir experiments/runs/main/transformer__full/seed0 --split test
"""
from __future__ import annotations
import argparse, io, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from src.datasets.har import ACTIVITIES


def compute_metrics(y_true, y_pred, n_classes=6) -> dict:
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(r.mean()),
        "per_class": {ACTIVITIES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                      "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def subject_bootstrap_ci(y_true, y_pred, subjects, n_boot=1000, seed=0, alpha=0.05):
    """95 % CI of macro-F1 by resampling SUBJECTS (not windows).
    Windows of one person are strongly correlated, so a window-level bootstrap
    would give over-optimistic (too narrow) intervals."""
    rng = np.random.RandomState(seed)
    subs = np.unique(subjects)
    by = {s: np.where(subjects == s)[0] for s in subs}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[s] for s in rng.choice(subs, len(subs), replace=True)])
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro",
                             labels=np.unique(y_true[idx]), zero_division=0))
    return [float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2)))]


def predict_proba(model, X, device, bs=512):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(device)
            out.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.concatenate(out)


def count_params(model) -> int:
    skip = ("recon_head", "mask_token")      # SSL-only parameters, not used at inference
    return int(sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith(skip)))


def model_size_mb(model) -> float:
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def measure_inference_ms(model, example, device, n=200, warmup=20) -> float:
    """Latency for ONE window (batch size 1), averaged over n runs."""
    import torch
    model.eval()
    x = example.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def evaluate_run(run_dir, split="test") -> dict:
    """Re-evaluate a saved run from its checkpoint (independent of train.py logging)."""
    import torch, joblib
    from src.datasets.har import get_splits
    from src.datasets.features import extract_features
    from src.models import build_model
    run_dir = Path(run_dir)
    res = json.load(open(run_dir / "result.json"))
    cfg, d = res["config"], res["config"]["data"]
    sp = get_splits(d["root"], d["channels"], d["window_len"], d["val_subjects"], d["split_seed"])
    X, y = getattr(sp, f"X_{split}"), getattr(sp, f"y_{split}")
    if cfg["model"]["name"] in ("rf", "svm"):
        pred = joblib.load(run_dir / "model.joblib").predict(extract_features(X))
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = build_model(cfg["model"]["name"], X.shape[-1], 6, **cfg["model"]["params"]).to(dev)
        m.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
        pred = predict_proba(m, X, dev).argmax(1)
    return compute_metrics(y, pred)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    m = evaluate_run(a.run_dir, a.split)
    print(f"[{a.split}] macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")
    for k, v in m["per_class"].items():
        print(f"  {k:<11} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")
