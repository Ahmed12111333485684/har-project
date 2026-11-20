"""Small shared helpers: seeding, config loading/merging, hardware info."""
from __future__ import annotations
import copy, json, os, platform, random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed every RNG we use so a (config, seed) pair is reproducible."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(cfg: dict, overrides) -> dict:
    """overrides: iterable of 'a.b.c=value' strings or a dict {'a.b.c': value}."""
    cfg = copy.deepcopy(cfg)
    items = overrides.items() if isinstance(overrides, dict) else (
        o.split("=", 1) for o in overrides)
    for k, v in items:
        if isinstance(v, str):
            v = yaml.safe_load(v)
        set_by_path(cfg, k, v)
    return cfg


def load_config(path, use_best: bool = True) -> dict:
    """default.yaml  <-  model config  <-  configs/best/<name>.yaml (tuned values)."""
    default = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    cfg = deep_merge(default, yaml.safe_load(open(path)))
    cfg.setdefault("name", Path(path).stem)
    best = ROOT / "configs" / "best" / f"{Path(path).stem}.yaml"
    if use_best and best.exists():
        cfg = apply_overrides(cfg, yaml.safe_load(open(best)) or {})
    return cfg


def hardware_string() -> str:
    import torch
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)} | CPU: {cpu} ({os.cpu_count()} threads)"
    return f"CPU: {cpu} ({os.cpu_count()} threads), no GPU"


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
