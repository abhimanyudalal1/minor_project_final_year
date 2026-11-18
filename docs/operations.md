# Running and auditing experiments

## Installation and commands

Create a project-local virtual environment and install `.[test]`. The CLI is
`burgers`; `python -m burgers_research.cli` is equivalent. Each stage accepts
`--config` and `--work-dir`. Defaults are `configs/study.yaml` and
`artifacts/study`. Every run records exact installed dependencies; the tested
Mac environment is also captured in `requirements-macos.lock.txt`.

```bash
burgers doctor
burgers matrix
burgers verify-solver
burgers benchmark
burgers generate-data
burgers validate-data
burgers train
burgers evaluate
burgers summarize
burgers build-report
```

`train` and `evaluate` process all runs by default; `--run fno-s0` selects one.
`matrix` lists registered names. Training must finish for the entire matrix
before evaluation locks validation selection and opens test data. Summarization
requires every evaluation and control. Use a new work directory for a changed
scientific protocol. Concurrent writers to one work directory are unsupported.

For integration verification, pass `--config configs/fixture.yaml --work-dir
artifacts/fixture` at every stage. Fixture outputs never substitute for production.

## Hardware and recovery

For unattended sequential execution, use the persistent worker:

```bash
python scripts/production.py start --device mps
python scripts/production.py status
python scripts/production.py stop
```

The worker freezes package source, configuration, documentation, and tests under
`artifacts/study/execution/source`. It uses that snapshot for every stage, keeps
per-stage logs and a heartbeat in `execution/status.json`, and stops on failure.
An exclusive lock prevents duplicate workers. Repeating `start` resumes at the
unfinished stage; generation and training reuse completed shards/checkpoints.
On macOS it prevents idle system sleep while running, but closing the lid,
rebooting, or terminating the process can interrupt it. Keep the machine powered
and awake for continuous work. Resume after interruptions using `start`.

The HDF5 reader retains up to 128 open shards with 1 MiB chunk caches per opened
grid dataset, avoiding repeated file opens for randomly shuffled training pairs.
Its memory stays bounded; neither this cache nor microbatching drops examples.

Use `--device mps`, `--device cpu`, or `--device cuda`; auto prefers CUDA then MPS.
On CUDA install an appropriate PyTorch wheel, then this package, and run doctor.
The scientific configuration is unchanged across devices. `--microbatch 16`
reduces memory while retaining effective batch 256 and every training example.

Atomic `last.pt` checkpoints occur every 100 updates and each epoch; `best.pt`
stores the best validation model. Resume by repeating the command. Recovery
includes optimizer/scheduler, epoch permutation cursor, partial loss sums,
normalization, and Python/NumPy/Torch/accelerator RNG states. Interrupted
accumulation groups replay from the last complete checkpoint.

Checkpoints contain serialized Python objects; only load trusted project files.
Source hashes reject implementation changes during a run. Model values transfer
across backends, but bit-identical cross-device results are not promised.
Sessions record backend changes; compare timings on matching hardware.

Colab/Kaggle can install the package, attach the same artifact directory, and
resume with `--device cuda`. Copy complete shards, manifests, and checkpoints to
persistent storage between sessions. There are no account-specific paths or
automatic uploads.

## Resource requirements

The raw reference archive is approximately 36.2 GB; derived grids add 4 GB.
The doctor calculates space from array shapes, checkpoint sizes, a conservative
JSON-metrics allowance, HDF5 overhead, one temporary shard/checkpoint, and a
5 GB working margin. Existing artifacts are credited when resuming. The former
fixed 100 GB reserve was unnecessarily restrictive and has been removed.
Generation and loading have bounded memory usage. The target CUDA device
has at least 16 GB VRAM; an M4 with 16 GB unified memory is supported subject to
the actual backend checks and runtime measurements.

Benchmarking uses a dedicated synthetic trajectory and full-architecture update
sizes. Estimates cover generation and training/validation, excluding I/O,
checkpointing, final evaluation, timestep refinement, and thermal throttling.
These measurements do not constitute a reduced scientific experiment.

## Verification record

See [verification-status.md](verification-status.md) for measured solver errors,
CPU/Metal checks, the completed fixture report, actual benchmark estimates, and
remaining production work. `python scripts/verify_fixture.py` exercises the CLI;
`bash scripts/run_study.sh` executes the full production sequence without reducing
the dataset or run matrix.

## Artifact index

- `verification/`: solver convergence, integrity, hardware parity, benchmark.
- `data/`: splits, HDF5 shards, checksum sidecars, manifest and normalization.
- `runs/<name>/`: checkpoints, histories, run results, session environments.
- `selection.json`: validation-only lambda selection locked before testing.
- `evaluation/`: per-trajectory metrics, curves, failures, controls.
- `summary/`: full sensitivity table and paired effects.
- `report/`: generated Markdown, figures, provenance hashes.

Null metrics mean failed/unavailable, not zero. Reference controls measure
residual floors; oracle interpolation controls measure representation error.
The generated report presents measurements and interpretation constraints.
Research discussion must follow production results, never fixture outcomes.
