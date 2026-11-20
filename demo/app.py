"""
Live demo - Wearable Human Activity Recognition (UCI-HAR)

Shows the WHOLE pipeline on real input:
  raw sensor signal -> sliding windows -> normalisation -> model -> prediction
  + 3D motion path and 3D phone orientation + what the model attends to
  + all four trained models side by side.

Run from the project root:   python demo/app.py
Needs the trained runs from:  python -m src.run_experiments final
"""
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                   # config paths like data/raw are relative

import gradio as gr                              # noqa: E402
from src import evaluate as ev                   # noqa: E402
from src.datasets.features import extract_features   # noqa: E402
from src.datasets.har import ACTIVITIES, CHANNEL_NAMES, get_splits, load_official  # noqa: E402
from src.models import build_model               # noqa: E402

FS = 50                                          # sampling rate the models were trained on
WIN, STRIDE = 128, 64                            # 2.56 s windows, 50 % overlap
MODEL_NAMES = ["resnet1d", "cnn_bilstm", "transformer", "rf"]
PRETTY = {"resnet1d": "ResNet-1D (advanced)", "cnn_bilstm": "CNN-BiLSTM (classic)",
          "transformer": "Transformer (modern)", "rf": "Random Forest (baseline)"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]
LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=50, b=40))

MEAN = STD = RAW_TEST = None
_models = {}


# ----------------------------------------------------------------------------- loading
def ensure_data():
    global MEAN, STD, RAW_TEST
    if MEAN is None:
        sp = get_splits("data/raw")              # same normalisation statistics as training
        MEAN, STD = sp.mean.astype(np.float32), sp.std.astype(np.float32)
        RAW_TEST = load_official("data/raw")["test"]


def run_dir(name, seed=0):
    return ROOT / "experiments" / "runs" / "main" / f"{name}__full" / f"seed{seed}"


def available_models():
    out = []
    for n in MODEL_NAMES:
        d = run_dir(n)
        if (d / "result.json").exists() and ((d / "model.pt").exists() or (d / "model.joblib").exists()):
            out.append(n)
    return out


def get_model(name):
    if name in _models:
        return _models[name]
    d = run_dir(name)
    cfg = json.load(open(d / "result.json"))["config"]
    if name == "rf":
        m = joblib.load(d / "model.joblib")
    else:
        m = build_model(cfg["model"]["name"], 9, 6, **cfg["model"]["params"])
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        m.eval()
    _models[name] = m
    return m


def predict(name, Wn):
    m = get_model(name)
    if name == "rf":
        return m.predict_proba(extract_features(Wn))
    return ev.predict_proba(m, Wn, torch.device("cpu"))


# ----------------------------------------------------------------------------- input
def test_session(subject, seed):
    """One real window of each activity from one TEST subject, concatenated (15 s)."""
    X, y, s = RAW_TEST
    rng = np.random.RandomState(int(seed))
    segs, truth = [], []
    for a in range(6):
        idx = np.where((s == subject) & (y == a))[0]
        segs.append(X[rng.choice(idx)])
        truth += [a] * WIN
    return np.concatenate(segs, 0).astype(np.float32), np.array(truth)


def _find(cols, pats):
    for p in pats:
        for c in cols:
            if re.search(p, c.lower()):
                return c
    return None


def parse_csv(path, fs_default):
    """Phyphox-style or plain CSV -> (N, 9) array in the model's channel order
    [total_acc xyz (g), body_acc xyz (g), body_gyro xyz (rad/s)] at 50 Hz."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    acc = [_find(cols, [rf"acc\w*[\s_\(]*{a}\b", rf"^a_?{a}$"]) for a in "xyz"]
    gyr = [_find(cols, [rf"gyr\w*[\s_\(]*{a}\b", rf"^g_?{a}$"]) for a in "xyz"]
    if None in acc or None in gyr:
        raise gr.Error("Could not find accelerometer (x,y,z) and gyroscope (x,y,z) columns. "
                       "Use names like 'Acceleration x (m/s^2)' / 'Gyroscope x (rad/s)' or ax,ay,az,gx,gy,gz.")
    tcol = _find(cols, [r"time"])
    A = df[acc].to_numpy(float)
    G = df[gyr].to_numpy(float)
    if tcol is not None:                         # resample to a uniform 50 Hz grid
        t = df[tcol].to_numpy(float)
        t = t - t[0]
        grid = np.arange(0, t[-1], 1 / FS)
        A = np.stack([np.interp(grid, t, A[:, i]) for i in range(3)], 1)
        G = np.stack([np.interp(grid, t, G[:, i]) for i in range(3)], 1)
    elif abs(fs_default - FS) > 1e-6:
        n = int(len(A) * FS / fs_default)
        g = np.linspace(0, len(A) - 1, n)
        A = np.stack([np.interp(g, np.arange(len(A)), A[:, i]) for i in range(3)], 1)
        G = np.stack([np.interp(g, np.arange(len(G)), G[:, i]) for i in range(3)], 1)
    if np.median(np.linalg.norm(A, axis=1)) > 3:     # m/s^2 -> g (UCI uses g)
        A = A / 9.80665
    if "deg" in str(gyr[0]).lower():
        G = np.deg2rad(G)
    if len(A) < WIN + 40:
        raise gr.Error(f"Signal too short ({len(A)/FS:.1f} s). Need at least {(WIN+40)/FS:.1f} s.")
    b, a = butter(3, 0.3 / (FS / 2), "low")          # gravity = low-pass, like the dataset authors
    grav = filtfilt(b, a, A, axis=0)
    return np.concatenate([A, A - grav, G], 1).astype(np.float32)


def make_windows(sig):
    starts = list(range(0, len(sig) - WIN + 1, STRIDE))
    return np.stack([sig[s:s + WIN] for s in starts]), starts


# ----------------------------------------------------------------------------- figures
def fig_raw(sig, truth):
    t = np.arange(len(sig)) / FS
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Accelerometer incl. gravity (g)", "Body acceleration (g)",
                                        "Gyroscope (rad/s)"))
    for r in range(3):
        for k, ax in enumerate("xyz"):
            fig.add_trace(go.Scatter(x=t, y=sig[:, 3 * r + k], name=ax, legendgroup=ax,
                                     showlegend=(r == 0), line=dict(width=1, color=COLORS[k])), row=r + 1, col=1)
    if truth is not None:                              # ground-truth segments
        i = 0
        while i < len(truth):
            j = i
            while j < len(truth) and truth[j] == truth[i]:
                j += 1
            fig.add_vrect(x0=i / FS, x1=j / FS, fillcolor=COLORS[truth[i]], opacity=0.10, line_width=0)
            fig.add_annotation(x=(i + j) / 2 / FS, y=1.04, yref="paper", text=ACTIVITIES[truth[i]],
                               showarrow=False, font=dict(size=11))
            i = j
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_layout(height=620, title="Step 1 - raw sensor signal", **LAYOUT)
    return fig


def fig_windows(sig, starts):
    t = np.arange(len(sig)) / FS
    mag = np.linalg.norm(sig[:, 3:6], axis=1)
    fig = go.Figure(go.Scatter(x=t, y=mag, line=dict(width=1, color="#444"), name="|body acc|"))
    for n, s in enumerate(starts[:80]):
        fig.add_vrect(x0=s / FS, x1=(s + WIN) / FS, fillcolor=COLORS[n % 2 * 3], opacity=0.07, line_width=0)
    fig.update_layout(height=330, title=f"Step 2 - sliding windows: {len(starts)} windows x {WIN} samples "
                                        f"(2.56 s), stride {STRIDE} (50 % overlap)",
                      xaxis_title="time (s)", yaxis_title="|body acc| (g)", **LAYOUT)
    return fig


def fig_timeline(st):
    probs, starts = st["probs"], st["starts"]
    c = (np.array(starts) + WIN / 2) / FS
    fig = go.Figure(go.Heatmap(z=probs.T, x=c, y=ACTIVITIES, zmin=0, zmax=1, colorscale="Blues",
                               colorbar=dict(title="P")))
    pred = probs.argmax(1)
    fig.add_trace(go.Scatter(x=c, y=[ACTIVITIES[p] for p in pred], mode="markers", name="prediction",
                             marker=dict(color="red", size=11, symbol="diamond")))
    if st["truth_w"] is not None:
        fig.add_trace(go.Scatter(x=c, y=[ACTIVITIES[p] for p in st["truth_w"]], mode="markers",
                                 name="ground truth", marker=dict(color="black", size=8, symbol="x")))
    fig.update_layout(height=380, title=f"Step 4 - {PRETTY[st['model']]}: probability per window",
                      xaxis_title="window centre (s)", **LAYOUT)
    return fig


def fig_norm(Wn_i):
    fig = go.Figure(go.Heatmap(z=Wn_i.T, y=CHANNEL_NAMES, x=np.arange(WIN) / FS, colorscale="RdBu",
                               zmid=0, colorbar=dict(title="z-score")))
    fig.update_layout(height=420, title="Step 3 - selected window after z-score normalisation "
                                        "(train-set mean/std per channel)", xaxis_title="time (s)", **LAYOUT)
    return fig


def rot_align(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.9999:
        ax = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = ax - a * (a @ ax)
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def phone_mesh(R):
    w, h, t = 0.5, 1.0, 0.08
    v = np.array([[sx * w / 2, sy * h / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T
    quads = [(0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6)]
    i, j, k = [], [], []
    for p0, p1, p2, p3 in quads:
        i += [p0, p0]; j += [p1, p2]; k += [p2, p3]
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=i, j=j, k=k, color="#4c78a8", opacity=0.9,
                     flatshading=True, name="phone")


def fig_3d(W_i, label):
    acc = W_i[:, 0:3]
    t = np.arange(WIN) / FS
    g = acc.mean(0)
    tilt = float(np.degrees(np.arccos(np.clip(g[2] / (np.linalg.norm(g) + 1e-9), -1, 1))))
    R = rot_align(g, np.array([0.0, 0.0, 1.0]))
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Accelerometer path (x, y, z) - colour = time",
                                        f"Phone orientation from gravity (tilt {tilt:.0f} deg)"))
    fig.add_trace(go.Scatter3d(x=acc[:, 0], y=acc[:, 1], z=acc[:, 2], mode="lines+markers",
                               marker=dict(size=3, color=t, colorscale="Viridis", showscale=True,
                                           colorbar=dict(title="s", x=0.45, len=0.6)),
                               line=dict(width=3, color="#888"), name="acc path"), row=1, col=1)
    fig.add_trace(phone_mesh(R), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.3], mode="lines+text", text=["", "up"],
                               line=dict(color="red", width=6), name="gravity reading",
                               showlegend=False), row=1, col=2)
    cam = dict(eye=dict(x=1.5, y=1.5, z=0.9))
    ax = dict(showbackground=False)
    fig.update_layout(height=480, title=f"Step 5 - 3D view of the selected window (class: {label})",
                      scene=dict(xaxis_title="x (g)", yaxis_title="y (g)", zaxis_title="z (g)", camera=cam),
                      scene2=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", camera=cam,
                                  xaxis_range=[-1, 1], yaxis_range=[-1, 1], zaxis_range=[-1, 1.4]),
                      **LAYOUT)
    return fig


def fig_attention(name, Wn_i, W_i):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                        subplot_titles=("Signal (accelerometer, g)", "Where the model looked"))
    t = np.arange(WIN) / FS
    for k, a in enumerate("xyz"):
        fig.add_trace(go.Scatter(x=t, y=W_i[:, k], name=a, line=dict(width=1, color=COLORS[k])), row=1, col=1)
    imp = None
    x = torch.from_numpy(Wn_i[None]).float()
    if name == "transformer":
        m = get_model(name)
        for b in m.blocks:
            if b.mixer is not None:
                b.mixer.keep_attn = True
        with torch.no_grad():
            m(x)
        last = m.blocks[-1].mixer
        if last is not None and last.last_attn is not None:
            cls = last.last_attn[0].mean(0)[0, 1:].numpy()           # CLS -> patches, avg over heads
            imp = np.repeat(cls, m.patch)
            imp = np.pad(imp, (0, WIN - len(imp)))
        for b in m.blocks:
            if b.mixer is not None:
                b.mixer.keep_attn = False
        title = "Transformer: last-layer CLS attention over the 8-sample patches"
    elif name == "resnet1d":
        m = get_model(name)
        with torch.no_grad():
            m(x)
        w = m.last_pool_weights[0, :, 0].numpy()
        imp = np.interp(np.arange(WIN), np.linspace(0, WIN - 1, len(w)), w)
        title = "ResNet-1D: temporal attention-pooling weights"
    else:
        title = "Attention is only available for the Transformer and ResNet-1D"
    if imp is not None:
        fig.add_trace(go.Bar(x=t, y=imp, marker_color="#d62728", name="importance"), row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(height=470, title=f"Step 6 - explainability. {title}", **LAYOUT)
    return fig


def fig_compare(Wn_i, names):
    Z, txt = [], []
    for n in names:
        p = predict(n, Wn_i[None])[0]
        Z.append(p)
        txt.append([f"{v:.2f}" for v in p])
    fig = go.Figure(go.Heatmap(z=Z, x=ACTIVITIES, y=[PRETTY[n] for n in names], zmin=0, zmax=1,
                               colorscale="Blues", text=txt, texttemplate="%{text}", colorbar=dict(title="P")))
    fig.update_layout(height=120 + 70 * len(names), title="Step 7 - all trained models on the same window",
                      **LAYOUT)
    return fig


# ----------------------------------------------------------------------------- callbacks
def analyze(source, subject, file, fs, model_name, session_seed):
    ensure_data()
    names = available_models()
    if not names:
        raise gr.Error("No trained models found. Run: python -m src.run_experiments final")
    if model_name not in names:
        model_name = names[0]
    if source.startswith("Test"):
        sig, truth = test_session(int(subject), session_seed)
    else:
        if file is None:
            raise gr.Error("Please upload a CSV file first.")
        sig, truth = parse_csv(file if isinstance(file, str) else file.name, float(fs)), None
    W, starts = make_windows(sig)
    Wn = ((W - MEAN) / STD).astype(np.float32)
    probs = predict(model_name, Wn)
    pred = probs.argmax(1)
    truth_w = None if truth is None else np.array([np.bincount(truth[s:s + WIN]).argmax() for s in starts])
    st = dict(sig=sig, W=W, Wn=Wn, starts=starts, model=model_name, probs=probs, truth_w=truth_w, names=names)
    counts = np.bincount(pred, minlength=6)
    md = f"### {PRETTY[model_name]}: most frequent activity = **{ACTIVITIES[int(counts.argmax())]}** " \
         f"({counts.max()}/{len(pred)} windows)\n\n"
    if truth_w is not None:
        md += f"Window accuracy on this simulated session: **{(pred == truth_w).mean() * 100:.0f}%** " \
              f"(subject {subject} is a TEST subject: never seen in training).\n\n"
    md += "Per-window predictions: " + ", ".join(ACTIVITIES[p] for p in pred[:12]) + (" ..." if len(pred) > 12 else "")
    return (st, fig_raw(sig, truth), fig_windows(sig, starts), fig_timeline(st), md,
            gr.update(minimum=0, maximum=len(starts) - 1, value=0, step=1))


def show_window(st, idx):
    if st is None:
        return None, None, None, None
    i = int(idx)
    p = st["probs"][i]
    label = ACTIVITIES[int(p.argmax())] + f" ({p.max() * 100:.0f}%)"
    return (fig_norm(st["Wn"][i]), fig_3d(st["W"][i], label),
            fig_attention(st["model"], st["Wn"][i], st["W"][i]), fig_compare(st["Wn"][i], st["names"]))


def build():
    with gr.Blocks(title="HAR live demo") as demo:
        gr.Markdown("# Wearable Human Activity Recognition - live demo\n"
                    "Smartphone inertial sensors -> 6 activities. UCI-HAR, subject-wise split, "
                    "models trained by our team (AI403).")
        st = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(["Test session (real data, unseen subject)", "Upload sensor CSV"],
                                  value="Test session (real data, unseen subject)", label="Input")
                subject = gr.Dropdown(TEST_SUBJECTS, value=TEST_SUBJECTS[0], label="Test subject")
                seed = gr.Slider(0, 20, value=0, step=1, label="Session seed (which windows are picked)")
                file = gr.File(label="CSV (Phyphox export: time, acc x/y/z, gyro x/y/z)", type="filepath")
                fs = gr.Slider(10, 200, value=50, step=1, label="Sampling rate (Hz) if CSV has no time column")
                model = gr.Dropdown(MODEL_NAMES, value="resnet1d", label="Model")
                btn = gr.Button("Run", variant="primary")
                summary = gr.Markdown()
                win = gr.Slider(0, 1, value=0, step=1, label="Window to inspect (steps 3, 5, 6, 7)")
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("1 Raw signal"):
                        p1 = gr.Plot()
                    with gr.Tab("2 Windowing"):
                        p2 = gr.Plot()
                    with gr.Tab("3 Normalisation"):
                        p3 = gr.Plot()
                    with gr.Tab("4 Prediction"):
                        p4 = gr.Plot()
                    with gr.Tab("5 3D view"):
                        p5 = gr.Plot()
                    with gr.Tab("6 Attention"):
                        p6 = gr.Plot()
                    with gr.Tab("7 Compare models"):
                        p7 = gr.Plot()
        gr.Markdown("*Uploaded phone data differs from the training setup (waist-mounted phone, 50 Hz): "
                    "expect domain shift, especially for Sitting vs Standing.*")
        btn.click(analyze, [source, subject, file, fs, model, seed], [st, p1, p2, p4, summary, win]) \
           .then(show_window, [st, win], [p3, p5, p6, p7])
        win.release(show_window, [st, win], [p3, p5, p6, p7])
    return demo


if __name__ == "__main__":
    build().launch()