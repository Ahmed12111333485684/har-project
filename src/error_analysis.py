"""
Error analysis for the main models (needs `run_experiments final` first).
  python -m src.error_analysis --seed 0
Produces in experiments/error_analysis/:
  confusion_<model>.png          normalised confusion matrices
  table6_errors_<model>.csv      error type | count | example windows | likely cause
  per_subject_f1.csv             macro-F1 per test person (who is hard?)
  examples_<model>.png           raw signals of misclassified windows
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from src.datasets.har import ACTIVITIES, CHANNEL_NAMES, load_official

# Hypotheses to be CHECKED against the example plots (they are explanations, not proofs).
CAUSES = {
    frozenset(["Sitting", "Standing"]): "Both are static; phone on the waist has near-identical motion. Only the gravity direction (tilt) differs and it varies per person/posture.",
    frozenset(["Walking", "Upstairs"]): "Same gait rhythm; stairs differ mainly by subtle vertical-acceleration amplitude.",
    frozenset(["Walking", "Downstairs"]): "Same gait rhythm; downstairs has sharper impact peaks that vary with speed and subject.",
    frozenset(["Upstairs", "Downstairs"]): "Both stair activities share periodicity; separation depends on small amplitude/phase cues.",
    frozenset(["Laying", "Sitting"]): "Occasional ambiguous orientation when the person reclines while sitting.",
    frozenset(["Laying", "Standing"]): "Rare; usually a posture transition inside the window.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="experiments")
    a = ap.parse_args()
    out = Path(a.root) / "error_analysis"; out.mkdir(parents=True, exist_ok=True)
    raw = None
    subj_rows = []
    for rd in sorted(Path(a.root, "runs", "main").glob(f"*__full/seed{a.seed}")):
        res = json.load(open(rd / "result.json")); model = res["model"]
        z = np.load(rd / "preds.npz")
        if "test_probs" not in z:
            continue
        y, pred, subj = z["test_y"], z["test_probs"].argmax(1), z["test_subj"]
        # ---- confusion matrix
        cm = confusion_matrix(y, pred, labels=range(6)); cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.8)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6), ACTIVITIES, rotation=45, ha="right"); ax.set_yticks(range(6), ACTIVITIES)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > .5 else "black")
        ax.set(xlabel="Predicted", ylabel="True", title=f"{model} (seed {a.seed}) - test")
        fig.tight_layout(); fig.savefig(out / f"confusion_{model}.png", dpi=150); plt.close(fig)
        # ---- error table (Table 6)
        rows = []
        for i in range(6):
            for j in range(6):
                if i != j and cm[i, j] > 0:
                    idx = np.where((y == i) & (pred == j))[0]
                    rows.append({"error type": f"{ACTIVITIES[i]} -> {ACTIVITIES[j]}", "count": int(cm[i, j]),
                                 "% of true class": round(100 * cmn[i, j], 1),
                                 "example (test window idx / subject)": f"{idx[0]} / S{subj[idx[0]]}",
                                 "likely cause (hypothesis)": CAUSES.get(frozenset([ACTIVITIES[i], ACTIVITIES[j]]),
                                                                          "unclear - inspect example plot"),
                                 "_idx": idx})
        tab = pd.DataFrame(rows).sort_values("count", ascending=False)
        tab.drop(columns="_idx").to_csv(out / f"table6_errors_{model}.csv", index=False)
        # ---- per-subject F1
        for s in np.unique(subj):
            m = subj == s
            subj_rows.append({"model": model, "subject": int(s),
                              "macro_f1": f1_score(y[m], pred[m], average="macro", labels=np.unique(y[m]), zero_division=0),
                              "errors": int((y[m] != pred[m]).sum()), "n": int(m.sum())})
        # ---- example plots of the top-3 error types
        if raw is None:
            raw = load_official(res["config"]["data"]["root"])["test"][0]
        top = tab.head(3)
        if len(top):
            fig, axs = plt.subplots(len(top), 2, figsize=(10, 2.6 * len(top)), squeeze=False)
            for r, (_, row) in enumerate(top.iterrows()):
                w = row["_idx"][0]
                for c, (sl, nm) in enumerate([(slice(0, 3), "total_acc"), (slice(6, 9), "body_gyro")]):
                    for k in range(3):
                        axs[r, c].plot(raw[w][:, sl][:, k], label=CHANNEL_NAMES[sl][k][-1], lw=1)
                    axs[r, c].set_title(f"{row['error type']} | {nm} | window {w}", fontsize=9)
                axs[r, 0].legend(fontsize=7, ncol=3)
            fig.tight_layout(); fig.savefig(out / f"examples_{model}.png", dpi=130); plt.close(fig)
        print(f"{model}: {int((y != pred).sum())} errors / {len(y)}; top: ",
              "; ".join(f"{r['error type']} ({r['count']})" for _, r in tab.head(3).iterrows()))
    if subj_rows:
        pd.DataFrame(subj_rows).pivot(index="subject", columns="model", values="macro_f1").round(3).to_csv(out / "per_subject_f1.csv")


if __name__ == "__main__":
    main()
