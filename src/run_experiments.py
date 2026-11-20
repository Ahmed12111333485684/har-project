"""
One entry point for every experiment of the project. Runs are RESUMABLE
(a finished run with result.json is skipped, so a Colab disconnect costs nothing).

  tune     same grid size for every model, seed 0, VALIDATION only  -> configs/best/<model>.yaml
  final    best config x seeds, evaluated on TEST                   -> group 'main'
  sweep    any grid:  --grid train.batch_size=32,64,128 --grid augment.names=[],[jitter]
  ablate   predefined component removals, evaluated on TEST         -> group 'ablation'

Global:  --set key=value ...   (e.g. --set train.epochs=2 for a smoke test)
         --out-root experiments  --models rf cnn_bilstm transformer resnet1d  --seeds 0 1 2
"""
from __future__ import annotations
import argparse, itertools, json, shutil
from pathlib import Path
import pandas as pd
import yaml
from src import train as T
from src.utils import ROOT, apply_overrides, load_config, set_seed

MODELS = ["rf", "cnn_bilstm", "transformer", "resnet1d"]
ABLATIONS = {   # variant name -> {override: value}; 'full' reference = group main
    "rf":         {"no_gyro": {"data.channels": "acc_only"}},
    "cnn_bilstm": {"no_gyro": {"data.channels": "acc_only"}, "no_aug": {"augment.names": []},
                   "no_bilstm": {"model.params.use_lstm": False}},
    "transformer": {"no_gyro": {"data.channels": "acc_only"}, "no_aug": {"augment.names": []},
                    "no_attention": {"model.params.mixer": "none"}},
    "resnet1d":   {"no_gyro": {"data.channels": "acc_only"}, "no_aug": {"augment.names": []},
                   "no_se": {"model.params.use_se": False},
                   "no_attn_pool": {"model.params.use_attn_pool": False}},
}


def cfg_for(model, extra, use_best=True):
    return apply_overrides(load_config(ROOT / "configs" / f"{model}.yaml", use_best=use_best), extra)


def run_once(cfg, seed, group, variant, out_root, test, force, verbose=False):
    out = Path(out_root) / "runs" / group / f"{cfg['name']}__{variant}" / f"seed{seed}"
    if (out / "result.json").exists() and not force:
        r = json.load(open(out / "result.json"))
        print(f"[skip] {group}/{cfg['name']}/{variant}/seed{seed}")
        return r
    r = T.run(cfg, seed, out, eval_test=test, verbose=verbose,
              meta={"group": group, "variant": variant})
    msg = f"val F1 {r['val']['macro_f1']:.4f}" + (f" | test F1 {r['test']['macro_f1']:.4f}" if test else "")
    print(f"[done] {group}/{cfg['name']}/{variant}/seed{seed}: {msg}  ({r['cost']['train_time_s']:.0f}s)", flush=True)
    return r


def cmd_tune(a):
    rows = []
    for m in a.models:
        base = cfg_for(m, a.set, use_best=False)
        grid = yaml.safe_load(open(ROOT / "configs" / f"{m}.yaml"))["tune_grid"]
        keys, vals = list(grid), list(grid.values())
        scores = []
        for combo in itertools.product(*vals):
            ov = dict(zip(keys, combo))
            variant = "_".join(f"{k.split('.')[-1]}={v}" for k, v in ov.items())
            r = run_once(apply_overrides(base, ov), 0, "tune", variant, a.out_root, False, a.force)
            scores.append((r["val"]["macro_f1"], ov))
            rows.append({"model": m, "variant": variant, "val_macro_f1": r["val"]["macro_f1"]})
        best = max(scores, key=lambda s: s[0])
        (ROOT / "configs" / "best").mkdir(exist_ok=True)
        yaml.safe_dump(best[1], open(ROOT / "configs" / "best" / f"{m}.yaml", "w"))
        print(f"==> best for {m}: {best[1]}  (val F1 {best[0]:.4f})")
    Path(a.out_root).mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(a.out_root) / "tuning_results.csv", index=False)


def cmd_final(a):
    for m in a.models:
        for s in a.seeds:
            run_once(cfg_for(m, a.set), s, "main", "full", a.out_root, True, a.force)

def _safe(v):
    import re
    s = json.dumps(v)
    return s if s == "[]" else re.sub(r'[^A-Za-z0-9_.]+', '-', s).strip('-')


def cmd_sweep(a):
    grid = {}
    for g in a.grid:
        k, v = g.split("=", 1)
        grid[k] = [yaml.safe_load(x) for x in _split_top(v)]
    for m in a.models:
        for combo in itertools.product(*grid.values()):
            ov = dict(zip(grid, combo))
            variant = "_".join(f"{k.split('.')[-1]}={_safe(v)}" for k, v in ov.items())
            variant = variant.replace("/", "-").replace(" ", "")
            for s in a.seeds:
                run_once(cfg_for(m, a.set + [f"{k}={json.dumps(v)}" for k, v in ov.items()]),
                         s, f"sweep_{a.tag}", variant, a.out_root, a.test, a.force)


def _split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def cmd_ablate(a):
    for m in a.models:
        for variant, ov in ABLATIONS[m].items():
            for s in a.seeds:
                run_once(cfg_for(m, a.set + [f"{k}={_safe(v)}" for k, v in ov.items()]),
                         s, "ablation", variant, a.out_root, True, a.force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "final", "sweep", "ablate"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", action="append", default=[])
    ap.add_argument("--tag", default="custom")
    ap.add_argument("--test", action="store_true", help="sweep: also evaluate on test (default: val only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-root", default="experiments")
    a = ap.parse_args()
    {"tune": cmd_tune, "final": cmd_final, "sweep": cmd_sweep, "ablate": cmd_ablate}[a.cmd](a)
