# Custom Burgers — 200 training trajectories, one seed (0)

This is a separate study using the original generated Burgers data. Its protocol
label is `custom-burgers-200trajectories-1seed-s0`; that label appears in the generated
report. It uses 200 training trajectories, matching the PDEBench pilot's training
count, with one model seed. It does not use PDEBench data.

| Setting | Value |
|---|---|
| Models | U-Net, FNO, physics-soft FNO, physics-constrained FNO |
| Registered runs | `unet-s0`, `fno-s0`, `physics-soft-s0`, `physics-constrained-s0` |
| Training trajectories actually used | 200 |
| Training subset selection | Fixed selection without replacement, subset seed 1729 |
| Model seed | 0 |
| Epochs | 60 per model |
| Effective batch / microbatch | 256 / 64 |
| Training pairs per epoch | 20,000 per model |
| Optimizer updates | 79 per epoch; 4,740 per model |
| Validation / test trajectories | Original 250 / 500 |
| Training grid and time | N=256, t≤1 |
| Evaluation | N=256/512/1024; 100 and 200 autoregressive steps |
| Equation viscosity | 0.01, as in the original custom study |

The configuration is copied from the completed original study's frozen config,
then changes the protocol label, model seeds, and training subset. The source
archive still contains 2,000 training trajectories, so `data.counts.train` remains
2,000 for archive validation. `training.train_trajectories: 200` controls actual
training. Normalization is recomputed using only the selected training trajectories
at N=256 through t=1. The saved training view records the exact IDs and its hash.

Models initialize from seed 0 and train from scratch. The workflow reuses the
validated source data, validates it again, registers the four runs, checks the
device, trains, evaluates, summarizes, and generates a report. It freezes its own
source/configuration and saves independent checkpoints and stage logs.

Compare this study with seed 0 of the original custom study to assess the effect
of reducing training data under the same 60-epoch schedule. This also reduces the
number of optimizer updates. Matching PDEBench's training count does not match
its viscosity, initial-condition distribution, or numerical source. One seed
cannot measure variation across training seeds; trajectory bootstrap intervals
are conditional on that seed.

## Inspect the run

- [Configuration](../configs/custom-200trajectories-seed0.yaml)
- [Live status](../artifacts/custom-200trajectories-seed0/execution/status.json)
- [Training log](../artifacts/custom-200trajectories-seed0/execution/train.log)
- [Training subset IDs and normalization](../artifacts/custom-200trajectories-seed0/training-view.json)
- [Registered models](../artifacts/custom-200trajectories-seed0/experiment-registry.json)
- [Frozen source](../artifacts/custom-200trajectories-seed0/execution/source/)
- [Report when complete](../artifacts/custom-200trajectories-seed0/report/report.md)
- [Original 2,000-trajectory custom report](../artifacts/eight-run/full/report/report.md)
- [PDEBench 200-trajectory, two-seed report](../artifacts/pdebench/report/report.md)

Start/resume from the project root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_revision.py run \
  --config configs/custom-200trajectories-seed0.yaml \
  --work-dir artifacts/custom-200trajectories-seed0 \
  --source-data-root artifacts/eight-run/full --device mps
```
