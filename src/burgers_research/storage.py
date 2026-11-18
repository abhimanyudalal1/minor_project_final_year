"""Transparent disk estimates; compression savings are not assumed."""
from pathlib import Path
import shutil

import torch

from .common import matrix
from .models import build_model, parameter_count


def storage_budget(cfg, root):
    root = Path(root)
    d = cfg["data"]
    snapshots = round(d["final_time"] / d["dt"]) + 1
    trajectories = sum(d["counts"].values())
    reference = trajectories * snapshots * d["reference_n"] * 8
    derived = trajectories * snapshots * sum(d["resolutions"]) * 4
    # Meta tensors count parameters without allocation or advancing the RNG.
    with torch.device("meta"):
        counts = {kind: parameter_count(build_model(kind, cfg)) for kind in ("fno", "unet")}
    runs = list(matrix(cfg))
    # Two checkpoints: float32 parameters + Adam's two moments, plus metadata.
    checkpoints = sum(24 * counts[r["model"]] + 131072 for r in runs)
    curves = 7 * (max(cfg["evaluation"]["horizons"]) + 1)
    adapters = sum(1 if n == d["base_n"] else 3 for n in d["resolutions"])
    records = d["counts"]["test"] * (len(runs) * adapters + 4 * len(d["resolutions"]))
    # Conservative allowance for pretty-printed JSON curves and per-record metadata.
    metrics = records * (40 * curves + 2048)
    data = reference + derived
    overhead = (data * 3 + 99) // 100  # HDF5 chunk indexes and metadata.
    temporary = min(d["shard_size"], trajectories) * snapshots * (
        d["reference_n"] * 8 + sum(d["resolutions"]) * 4)
    temporary += max(counts.values()) * 12 + 65536
    safety = int(cfg["runtime"].get("disk_safety_margin_gb", 5) * 1e9)
    components = {"data": data + overhead, "runs": checkpoints, "evaluation": metrics}
    existing = {}
    for directory, expected in components.items():
        folder = root / directory
        written = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink()) if folder.exists() else 0
        existing[directory] = min(written, expected)
    required = sum(components.values()) + temporary + safety - sum(existing.values())
    return {"reference_bytes": reference, "derived_bytes": derived,
            "checkpoint_allowance_bytes": checkpoints, "metrics_allowance_bytes": metrics,
            "hdf5_overhead_allowance_bytes": overhead, "temporary_allowance_bytes": temporary,
            "safety_margin_bytes": safety, "existing_bytes_credited": existing,
            "required_free_bytes": required, "free_bytes": shutil.disk_usage(root).free}


def check_storage(cfg, root):
    budget = storage_budget(cfg, root)
    if budget["free_bytes"] < budget["required_free_bytes"]:
        raise RuntimeError(
            f"Insufficient disk: {budget['free_bytes']/1e9:.2f} GB free; "
            f"estimated remaining study artifacts plus working margin need "
            f"{budget['required_free_bytes']/1e9:.2f} GB. Run doctor for the breakdown.")
    return budget
