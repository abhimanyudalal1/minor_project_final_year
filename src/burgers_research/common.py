from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomic JSON; replace nonfinite floats with null, preserving failure flags."""
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    d = cfg["data"]
    assert d["viscosity"] > 0 and d["dt"] > 0
    assert 0 < d["train_time"] < d["final_time"]
    assert d["base_n"] in d["resolutions"]
    assert d["ic_modes"] < min(d["resolutions"]) // 2
    assert all(n % 8 == 0 and n <= d["reference_n"] for n in d["resolutions"])
    assert set(d["counts"]) == {"train", "validation", "test"}
    if cfg.get("evaluation_only", False):
        assert d["counts"]["train"] == d["counts"]["validation"] == 0 and d["counts"]["test"] > 0
    else:
        assert all(v > 0 for v in d["counts"].values())
    assert cfg["runtime"]["microbatch"] > 0
    assert cfg["training"]["batch_size"] > 0
    assert cfg["training"]["epochs"] > 0
    for t in (d["final_time"], d["train_time"]):
        assert np.isclose(t / d["dt"], round(t / d["dt"]))
    assert max(cfg["evaluation"]["horizons"]) <= round(d["final_time"] / d["dt"])
    if "variants" in cfg["training"]:
        variants = cfg["training"]["variants"]
        assert variants and len({v["name"] for v in variants}) == len(variants)
        for v in variants:
            assert v["name"] and all(c.isalnum() or c in "-_" for c in v["name"])
            assert v["model"] in ("fno", "unet")
            assert v.get("stepping", "direct") in ("direct", "residual")
            assert v.get("physics_steps", 1) in (1, 2)
            assert v.get("weight", 0) >= 0
            assert v.get("derivative", "spectral") in ("spectral", "finite_difference")
        names = {v["name"] for v in variants}
        for a, b in cfg["evaluation"].get("comparisons", []):
            assert a in names and b in names and a != b
    assert cfg["training"].get("scheduler", "step") in ("step", "cosine")
    assert 0 <= cfg["training"].get("eta_min", 0) <= cfg["training"]["learning_rate"]
    if "train_trajectories" in cfg["training"]:
        assert 0 < cfg["training"]["train_trajectories"] <= d["counts"]["train"]
    return cfg


def data_signature(cfg):
    fields = {k: cfg[k] for k in ("protocol", "fixture", "data", "solver")}
    fields["protocol"] = cfg.get("dataset_protocol", cfg["protocol"])
    return digest(fields)


def experiment_signature(cfg):
    return digest({k: v for k, v in cfg.items() if k != "runtime"})


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "source_hash": source_hash(),
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "dependencies": subprocess.check_output(
                [os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}


def host_peak_memory_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value*1024)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run doctor and select CPU or CUDA explicitly.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def gradients_finite(model):
    # Reduce on device, synchronizing once instead of once per parameter tensor.
    checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    return bool(torch.stack(checks).all()) if checks else True


def matrix(cfg):
    if "variants" in cfg["training"]:
        for seed in cfg["training"]["seeds"]:
            for variant in cfg["training"]["variants"]:
                yield {**variant, "family": variant["name"], "name": f"{variant['name']}-s{seed}",
                       "seed": seed, "weight": float(variant.get("weight", 0)),
                       "derivative": variant.get("derivative", "spectral"),
                       "stepping": variant.get("stepping", "direct"),
                       "conserve_mean": variant.get("conserve_mean", False),
                       "physics_steps": variant.get("physics_steps", 1)}
        return
    for seed in cfg["training"]["seeds"]:
        yield {"name": f"unet-s{seed}", "model": "unet", "seed": seed, "weight": 0.0,
               "derivative": "spectral"}
        yield {"name": f"fno-s{seed}", "model": "fno", "seed": seed, "weight": 0.0,
               "derivative": "spectral"}
        for method in cfg["training"]["derivatives"]:
            for weight in cfg["training"]["weights"]:
                yield {"name": f"physics-{method}-l{weight:g}-s{seed}", "model": "fno",
                       "seed": seed, "weight": float(weight), "derivative": method}
