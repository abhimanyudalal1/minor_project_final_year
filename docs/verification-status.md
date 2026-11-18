# Implementation and verification status — 2026-11-14

## Completed

- Installable Python package and CLI covering the complete approved pipeline.
- Full 50-run registry and separate integration-fixture configuration.
- CPU float64 ETDRK4 reference solver, resumable HDF5 shards, split manifests,
  checksums, training-only normalization, and generated dataset cards.
- Standard periodic U-Net (671,457 real parameters) and FNO (549,505 real
  degrees of freedom), with the same FNO used for every physics-loss variant.
- Checkpointed training, validation-only selection, autoregressive evaluation,
  interpolation/reference controls, per-trajectory diagnostics, paired
  bootstrap statistics, and generated report/figures.
- **26 automated tests passed**. Production-sized architectures were exercised
  at N=256/512/1024. Tests include exact CPU restart equivalence and data
  corruption detection.
- The public CLI completed the entire fixture pipeline, including all eight
  fixture runs, controls, summaries, and report generation. These results are
  labelled **not research evidence**.

## Numerical evidence

Production solver verification used N=8192 versus 16384, four independent
development seeds, and internal timestep 0.001 versus 0.0005.

| Check | Observed maximum |
|---|---:|
| Normalized temporal difference | 5.873e-9 |
| Normalized spatial difference | 4.233e-16 |
| Normalized mass drift | 1.259e-16 |
| Positive energy increment | 0 |
| Analytic Cole–Hopf relative error | 1.045e-13 |

The approved tolerances passed without loosening or changing the grid. This is
verification on the development cases; each generated production trajectory
still receives its own physical checks.

## Hardware and measured budget

Environment: M4 MacBook Air, 16 GB unified memory, Python 3.14.7, PyTorch 2.15.0.
CPU and MPS checks passed for model outputs, physics-loss gradients, FFTs, and
complex-parameter Adam updates. Maximum prediction differences were below
4e-8. The execution sandbox hides Metal; direct GPU execution was used for
these checks. CUDA has not been tested on this machine.

The final benchmark uses ten measured effective updates after warmup per
case, microbatch 32, effective batch 256, and includes CPU-to-GPU transfers,
gradient checks, and logging synchronizations.

| Case | Estimated training + validation hours per run |
|---|---:|
| U-Net | 21.32 |
| Vanilla FNO | 6.53 |
| Spectral physics FNO | 7.43 |
| Finite-difference physics FNO | 6.73 |

The complete matrix projects to **422.4 hours**, approximately 17.6 days of
continuous execution. Serial reference generation projects to **1.25 hours**.
These are short benchmark extrapolations, not measured full-run durations.
They exclude HDF5 I/O, checkpointing, final evaluation, and sustained thermal
throttling. The earlier 290-hour estimate omitted some training-loop overhead;
this measurement supersedes it. No dataset size, seed count, or training budget
was reduced to obtain these estimates.

## Storage correction and remaining work

An earlier generation attempt stopped at an arbitrary 100 GB reserve. That
was an overly conservative policy, not an actual technical requirement. It has
been replaced by a calculated budget: approximately 36.2 GB of reference arrays,
4.0 GB of derived arrays, plus checkpoint/metrics allowances, HDF5 overhead,
temporary files, and a 5 GB working margin. The full study fits the current
approximately 71 GB free; the calculated budget is 57.9 GB, including a generous
10.6 GB allowance for pretty-printed metrics and the 5 GB margin. No dataset or
experiment reduction is needed. The storage regression tests verify both
adequate-capacity acceptance and genuine insufficient-capacity rejection.
Production execution has now been started with the full approved configuration.
The persistent worker's live stage, heartbeat, completed shard count, and finished
run count are recorded in `../artifacts/study/execution/status.json`. It proceeds
through generation, validation, all 50 training runs, evaluation, statistics, and
report generation, stopping if a stage fails. Starting the job is not completion
of those experiments; the live status and artifacts establish actual progress.

Run the unchanged scientific configuration using the calculated capacity check.
CUDA execution needs its own doctor check and
benchmark. The production experiments, final scientific results, and
results-based discussion remain outstanding until those stages finish; implementation verification is
not completion of the research study.

## Evidence files

- `../artifacts/study/verification/solver.json`
- `../artifacts/study/verification/doctor-mps.json`
- `../artifacts/study/verification/benchmark.json`
- `../artifacts/study/verification/storage.json`
- `../artifacts/study/execution/status.json`
- `../artifacts/study/execution/source/snapshot.json`
- `../artifacts/study/experiment-registry.json`
- `../artifacts/verified-fixture/report/report.md`
- `../requirements-macos.lock.txt`

Generated artifacts are excluded from version control by default. Retain these
files with the source snapshot when auditing or transferring the work.

## Persistent execution verification

The persistent worker completed all nine stages on a separate integration fixture,
including eight training runs and final report generation. The exclusive worker
lock, per-stage logs, source snapshot, and completion status are retained under
`../artifacts/runner-fixture/execution/`. The HDF5 reader now keeps a bounded cache
of open shards and datasets to avoid repeated opens during shuffled training;
its eviction path and checkpoint recovery passed the integration test. No
scientific configuration was changed by these execution improvements.
