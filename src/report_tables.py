"""Builds the report tables (mean ± SD over seeds) from experiments/runs/**/result.json.
  python -m src.report_tables      -> experiments/tables/*.csv and tables.md"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.datasets.har import ACTIVITIES

OUT = Path("experiments/tables")


def load(root="experiments"):
    rows = []
    for p in Path(root, "runs").rglob("result.json"):
        r = json.load(open(p))
        row = {"group": r["group"], "variant": r["variant"], "model": r["model"], "seed": r["seed"],
               "val_f1": r["val"]["macro_f1"], "val_acc": r["val"]["accuracy"],
               "epochs": r["epochs_run"], **{f"cost_{k}": v for k, v in r["cost"].items()}}
        if "test" in r:
            t = r["test"]
            row.update(test_f1=t["macro_f1"], test_acc=t["accuracy"], test_bal_acc=t["balanced_accuracy"],
                       ci_lo=t["macro_f1_ci95_subject_bootstrap"][0], ci_hi=t["macro_f1_ci95_subject_bootstrap"][1],
                       **{f"f1_{k}": v["f1"] for k, v in t["per_class"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def ms(s, d=3):
    s = s.dropna()
    return "n/a" if len(s) == 0 else f"{s.mean():.{d}f} ± {s.std(ddof=1) if len(s) > 1 else 0:.{d}f}"


def main(root="experiments"):
    global OUT
    OUT = Path(root) / "tables"
    df = load(root)
    if df.empty:
        print(f"No runs found under {root}/runs - run src.run_experiments first.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    md = []
    main_df = df[df.group == "main"]
    if len(main_df):
        g = main_df.groupby("model")
        t4 = pd.DataFrame({"n_seeds": g.size(), "Macro-F1 (test)": g.test_f1.apply(ms),
                           "Accuracy (test)": g.test_acc.apply(ms), "Balanced acc (test)": g.test_bal_acc.apply(ms),
                           "Macro-F1 (val)": g.val_f1.apply(ms),
                           "Subject-bootstrap 95% CI (mean of seeds)": g.apply(lambda x: f"[{x.ci_lo.mean():.3f}, {x.ci_hi.mean():.3f}]")})
        t4.to_csv(OUT / "table4_main_results.csv"); md += ["## Table 4 - main results (mean ± SD over seeds)", t4.to_markdown(), ""]
        pc = pd.DataFrame({a: g[f"f1_{a}"].apply(ms) for a in ACTIVITIES})
        pc.to_csv(OUT / "per_class_f1.csv"); md += ["## Per-class F1 (test)", pc.to_markdown(), ""]
        t7 = g.agg(params=("cost_params", "mean"), size_MB=("cost_size_mb", "mean"),
                   train_s=("cost_train_time_s", "mean"), infer_ms=("cost_inference_ms_per_window", "mean"),
                   hardware=("cost_hardware", "first")).round(3)
        t7.to_csv(OUT / "table7_cost.csv"); md += ["## Table 7 - computational cost", t7.to_markdown(), ""]
    ab = df[df.group == "ablation"]
    if len(ab) and len(main_df):
        rows = []
        for (m, v), x in ab.groupby(["model", "variant"]):
            full = main_df[main_df.model == m].test_f1
            rows.append({"model": m, "component removed": v, "Macro-F1 (test)": ms(x.test_f1),
                         "change vs full": f"{x.test_f1.mean() - full.mean():+.3f}",
                         "full model": ms(full), "params": int(x.cost_params.mean())})
        t5 = pd.DataFrame(rows); t5.to_csv(OUT / "table5_ablation.csv", index=False)
        md += ["## Table 5 - ablation", t5.to_markdown(index=False), ""]
    for grp, x in df[df.group.str.startswith("sweep_")].groupby("group"):
        s = x.groupby(["model", "variant"]).agg(val_f1=("val_f1", ms), seeds=("seed", "count")).reset_index()
        s.to_csv(OUT / f"{grp}.csv", index=False)
        md += [f"## {grp} (validation macro-F1)", s.to_markdown(index=False), ""]
    tu = df[df.group == "tune"][["model", "variant", "val_f1"]]
    if len(tu):
        md += ["## Tuning (seed 0, validation)", tu.to_markdown(index=False), ""]
    (OUT / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="experiments")
    main(ap.parse_args().root)
