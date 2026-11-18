"""Full dataset, three seed-0 runs, preliminary validation at epoch 10.

Run with PYTHONPATH pointing to the frozen production source snapshot. Checkpoints
retain the unchanged 500-epoch protocol and can later resume in the full matrix.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
from datetime import datetime, timezone

import torch

from burgers_research.common import load_config, matrix, write_json
from burgers_research.data import generate, validate
from burgers_research.training import train

ROOT = Path(__file__).resolve().parents[1]/"artifacts"/"study"
EXECUTION = ROOT/"execution"
STATUS = EXECUTION/"quick-status.json"
CONFIG = EXECUTION/"source"/"configs"/"production.yaml"


def update(status, stage, **extra):
    write_json(STATUS, {"status":status, "stage":stage, "pid":os.getpid(),
                        "updated_utc":datetime.now(timezone.utc).isoformat(), **extra})
    print(f"{status}: {stage}", flush=True)


def stop(signum, frame):
    raise KeyboardInterrupt


def main():
    EXECUTION.mkdir(parents=True, exist_ok=True)
    with open(EXECUTION/"worker.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        cfg = load_config(CONFIG)
        cfg["runtime"]["device"] = "mps"
        specs = [s for s in matrix(cfg) if s["seed"] == 0 and (
            (s["model"] in ("unet", "fno") and s["weight"] == 0) or
            (s["model"] == "fno" and s["derivative"] == "spectral" and s["weight"] == 1.0))]
        assert len(specs) == 3
        stage = "generate-data"
        try:
            update("running", stage)
            generate(cfg, ROOT)
            stage = "validate-data"
            update("running", stage)
            validate(cfg, ROOT)
            updates_per_epoch = math.ceil(cfg["data"]["counts"]["train"] *
                round(cfg["data"]["train_time"]/cfg["data"]["dt"]) / cfg["training"]["batch_size"])
            target = 10 * updates_per_epoch + 1
            rows = []
            for spec in specs:
                stage = spec["name"]
                update("running", stage, completed_runs=[r["name"] for r in rows])
                best = ROOT/"runs"/spec["name"]/"best.pt"
                if not best.exists() or torch.load(best, map_location="cpu", weights_only=False)["state"]["epoch"] < 10:
                    train(cfg, ROOT, spec, max_updates=target)
                checkpoint = torch.load(best, map_location="cpu", weights_only=False)
                rows.append({"name":spec["name"], "validation_rollout_error":checkpoint["state"]["best_score"],
                             "epoch":checkpoint["state"]["best_epoch"], "checkpoint":str(best)})
            folder = ROOT/"quick-results"
            folder.mkdir(exist_ok=True)
            write_json(folder/"validation.json", {"dataset_hash":checkpoint["dataset_hash"], "runs":rows,
                "limitations":"Preliminary 10-epoch validation, single seed, one physics setting; test set remains locked."})
            lines = ["# Preliminary validation results", "", "Full dataset; one seed; first 10 epochs. "
                     "These are validation results, not final test results or statistically replicated conclusions.", "",
                     "| Model | Epoch | Mean 100-step validation relative L2 |", "|---|---:|---:|"]
            lines += [f"| {r['name']} | {r['epoch']} | {r['validation_rollout_error']:.6g} |" for r in rows]
            (folder/"validation.md").write_text("\n".join(lines)+"\n")
            update("complete", "preliminary-validation", completed_runs=[r["name"] for r in rows])
        except KeyboardInterrupt:
            update("stopped", stage, message="Completed shards and checkpoints are preserved.")
            raise SystemExit(130)
        except Exception as exc:
            update("failed", stage, message=str(exc))
            raise


if __name__ == "__main__":
    main()
