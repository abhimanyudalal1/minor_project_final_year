# Physics-constrained neural operators for periodic Burgers' equation

For the current execution state, step-by-step workflow, metric formulas, and
artifact inspection commands, see [Current pipeline walkthrough](PIPELINE_WALKTHROUGH.md).

A reproducible research pipeline comparing a standard 1D U-Net, a vanilla
Fourier Neural Operator, and the identical FNO trained with a soft conservative
PDE-residual loss. The study measures autoregressive accuracy, mass drift,
energy dissipation, stability, and zero-shot spatial resolution transfer.

**Status:** the original custom-data epoch-26 pilots are complete. A separate
[v2 pipeline](docs/revision-v2.md) now implements a 12-run cosine-scheduled study,
matched-capacity FNOs, a constrained model, low-data ablation, and low-viscosity
evaluation. The [one-day execution](docs/one-day-run.md) is now running both training
budgets and the stress-data workflow with fixed hyperparameters. End-to-end software
fixtures verify the revised workflow. PDEBench integration remains pending.
See the [v1 walkthrough](PIPELINE_WALKTHROUGH.md) for prior results and metric definitions.
Storage is checked against calculated artifact sizes, not a fixed 100 GB reserve.
See the [verification record and measured compute budget](docs/verification-status.md).
The approved [plan](plan.md) and original
[proposal](docs/proposal-original.md) are preserved.

## Research protocol

The full study uses 2,000/250/500 train/validation/test trajectories, an 8,192-point
reference solver, training at N=256, testing at N=256/512/1024, and five seeds.
Two baselines plus eight physics configurations produce **50 training runs**.
The complete dataset is streamed from disk. Microbatch accumulation and resumable
checkpoints preserve study size when device memory is limited.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
burgers doctor
burgers matrix
burgers verify-solver
burgers benchmark
```

After reviewing numerical verification, storage availability, and the benchmark:

```bash
burgers generate-data
burgers validate-data
burgers train
burgers evaluate
burgers summarize
burgers build-report
```

All stages accept `--config configs/study.yaml --work-dir artifacts/study`.
Use `--device mps`, `--device cuda`, or `--device cpu` to select execution.
`--microbatch 16` changes memory usage while retaining the effective batch of 256.
`train --run fno-s0` resumes or executes a single registered run.

For software verification only, pass `--config configs/fixture.yaml` and
`--work-dir artifacts/fixture` to every stage. Fixture outputs are labelled and
must not be used as evidence for the research questions.

`python scripts/verify_fixture.py` runs all fixture stages. With an activated
environment, `bash scripts/run_study.sh` runs the full pipeline; set
`BURGERS_WORK_DIR` and `BURGERS_DEVICE` to choose storage and execution backend.

For persistent background execution, use `python scripts/production.py start
--device mps`; inspect it with `python scripts/production.py status`. The worker
uses a frozen source snapshot and stops on failure. See the operations guide for
stopping and resuming it.

## Documentation

- [Active one-day run: scope, deadline, and progress](docs/one-day-run.md)
- [Revised v2 protocol, implementation, and execution commands](docs/revision-v2.md)
- [Protocol, selection, statistics, and failures](docs/protocol.md)
- [Solver, data generation, storage, and residual derivation](docs/numerics-and-data.md)
- [Architecture definitions, diagrams, and tests](docs/architectures.md)
- [Execution, recovery, hardware requirements, and artifact index](docs/operations.md)
- [Research bibliography](docs/references.bib)

Reference solutions are validated numerical approximations. Soft residual
penalties do not imply exact conservation. Conclusions must follow the locked
test results, including negative findings and computational trade-offs.
