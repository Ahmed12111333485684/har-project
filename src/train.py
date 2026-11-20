"""
Train / fit ONE model for ONE seed and write everything to out_dir:
  result.json (metrics + cost + config), log.csv, model.pt|model.joblib, preds.npz
Usage:
  python -m src.train --config configs/transformer.yaml --seed 0 --out experiments/runs/demo
  python -m src.train --config configs/rf.yaml --set model.params.max_depth=10 --test
The TEST split is evaluated only when --test is given (final runs); tuning uses VALIDATION only.
"""
from __future__ import annotations
import argparse, copy, csv, io, math, pickle, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from src import evaluate as ev
from src.datasets.augment import build_augmenter
from src.datasets.features import extract_features
from src.datasets.har import HARDataset, get_splits
from src.models import SKLEARN_MODELS, build_model, build_sklearn
from src.utils import apply_overrides, hardware_string, load_config, save_json, set_seed


def class_weights(y, n=6):
    cnt = np.bincount(y, minlength=n)
    return torch.tensor(len(y) / (n * np.maximum(cnt, 1)), dtype=torch.float32)


def lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup)))


def _finish(res, sp, val_pred_probs, test_pred_probs, out, eval_test):
    """Metrics shared by DL and sklearn paths."""
    res["val"] = ev.compute_metrics(sp.y_val, val_pred_probs.argmax(1))
    save = {"val_probs": val_pred_probs, "val_y": sp.y_val, "val_subj": sp.s_val}
    if eval_test:
        pred = test_pred_probs.argmax(1)
        res["test"] = ev.compute_metrics(sp.y_test, pred)
        res["test"]["macro_f1_ci95_subject_bootstrap"] = ev.subject_bootstrap_ci(sp.y_test, pred, sp.s_test)
        save.update(test_probs=test_pred_probs, test_y=sp.y_test, test_subj=sp.s_test)
    np.savez_compressed(out / "preds.npz", **save)
    save_json(res, out / "result.json")
    return res


def run_sklearn(cfg, seed, out, eval_test, sp):
    import joblib
    m = cfg["model"]
    clf = build_sklearn(m["name"], seed=seed, **m["params"])
    Ftr, Fva = extract_features(sp.X_train), extract_features(sp.X_val)
    t0 = time.time(); clf.fit(Ftr, sp.y_train); train_time = time.time() - t0
    joblib.dump(clf, out / "model.joblib")
    est = clf[-1]
    n_params = (sum(t.tree_.node_count for t in est.estimators_) if m["name"] == "rf"
                else int(est.n_support_.sum() * Ftr.shape[1]))
    x1 = sp.X_val[:1]
    t = time.perf_counter()
    for _ in range(100):
        clf.predict(extract_features(x1))
    infer_ms = (time.perf_counter() - t) / 100 * 1000
    res = {"cost": {"params": int(n_params), "params_note": "tree nodes" if m["name"] == "rf" else "SV x features",
                    "size_mb": len(pickle.dumps(clf)) / 1e6, "train_time_s": train_time,
                    "inference_ms_per_window": infer_ms, "hardware": hardware_string()},
           "epochs_run": None, "best_epoch": None}
    te = clf.predict_proba(extract_features(sp.X_test)) if eval_test else None
    return _finish(res, sp, clf.predict_proba(Fva), te, out, eval_test)


def run_dl(cfg, seed, out, eval_test, sp, verbose):
    tc = cfg["train"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], sp.X_train.shape[-1], 6, **cfg["model"]["params"]).to(dev)
    if cfg.get("pretrained"):
        sd = torch.load(cfg["pretrained"], map_location=dev)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}     # fresh classifier head
        model.load_state_dict(sd, strict=False)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(HARDataset(sp.X_train, sp.y_train, build_augmenter(cfg["augment"]), seed),
                        batch_size=tc["batch_size"], shuffle=True, drop_last=True, generator=g)
    cw = class_weights(sp.y_train).to(dev) if tc["class_weights"] else None
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=tc["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_lambda(e, tc["warmup_epochs"], tc["epochs"]))

    best, best_state, best_ep, bad, rows = -1.0, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, tc["epochs"] + 1):
        model.train(); tot = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        vp = ev.predict_proba(model, sp.X_val, dev)
        m = ev.compute_metrics(sp.y_val, vp.argmax(1))
        rows.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_macro_f1": m["macro_f1"],
                     "val_acc": m["accuracy"], "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"  ep{ep:3d} loss {tot/max(n,1):.4f}  val F1 {m['macro_f1']:.4f}", flush=True)
        if m["macro_f1"] > best + 1e-4:
            best, best_ep, bad = m["macro_f1"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)                       # best-on-VALIDATION weights
    torch.save(best_state, out / "model.pt")
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

    example = torch.from_numpy(sp.X_val[:1]).float()
    res = {"cost": {"params": ev.count_params(model), "size_mb": ev.model_size_mb(model),
                    "train_time_s": train_time,
                    "inference_ms_per_window": ev.measure_inference_ms(model, example, dev),
                    "hardware": hardware_string()},
           "epochs_run": len(rows), "best_epoch": best_ep}
    te = ev.predict_proba(model, sp.X_test, dev) if eval_test else None
    return _finish(res, sp, ev.predict_proba(model, sp.X_val, dev), te, out, eval_test)


def run(cfg, seed, out_dir, eval_test=False, verbose=False, meta=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    d = cfg["data"]
    sp = get_splits(d["root"], d["channels"], d["window_len"], d["val_subjects"],
                    d["split_seed"], cfg.get("label_fraction", 1.0), label_seed=seed)
    if cfg["model"]["name"] in SKLEARN_MODELS:
        res = run_sklearn(cfg, seed, out, eval_test, sp)
    else:
        res = run_dl(cfg, seed, out, eval_test, sp, verbose)
    res.update({"model": cfg["model"]["name"], "seed": seed, "config": cfg,
                "n_train_windows": int(len(sp.y_train)),
                "split": {"train": sp.train_subjects, "val": sp.val_subjects, "test": sp.test_subjects},
                **(meta or {})})
    save_json(res, out / "result.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=0.0005")
    ap.add_argument("--test", action="store_true", help="also evaluate on the TEST split")
    ap.add_argument("--no-best", action="store_true", help="ignore configs/best/*.yaml")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config, use_best=not a.no_best), a.set)
    out = a.out or f"experiments/runs/adhoc/{cfg['name']}/seed{a.seed}"
    r = run(cfg, a.seed, out, eval_test=a.test, verbose=True)
    print("VAL  macro-F1 %.4f" % r["val"]["macro_f1"])
    if a.test:
        print("TEST macro-F1 %.4f  CI95 %s" % (r["test"]["macro_f1"], r["test"]["macro_f1_ci95_subject_bootstrap"]))
    print("cost:", r["cost"])
