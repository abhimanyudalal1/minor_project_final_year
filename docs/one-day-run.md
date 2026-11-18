# One-day execution: fixed custom-data experiments

Started **15 November 2026, 23:33 IST**. Deadline: **16 November 2026, 23:27 IST**,
24 hours after the user's instruction. No hyperparameter search is scheduled.

## Work being executed

The persistent controller is [one_day.py](../scripts/one_day.py). It runs a serial
Metal GPU queue and an independent CPU queue for numerical data generation.

1. **Low-data study:** 200 fixed training trajectories, four variants, three seeds,
   60 epochs each; validation-selected checkpoints; all 500 smooth test trajectories;
   resolutions 256/512/1024; 100/200-step rollouts; controls, bootstrap, and report.
2. **Full-data study:** same experiment using all 2,000 training trajectories,
   followed by the same complete test evaluation and report.
3. **Low-viscosity data:** independently verify the ν=0.001 solver, then generate
   and validate 300 test-only trajectories on the 8,192-point reference grid.
   This runs on CPU while the GPU queue proceeds.
4. **Shift evaluation:** evaluate both completed studies on all 300 low-viscosity
   trajectories at all three resolutions and both horizons, then generate summaries
   and reports.
5. Write a consolidated execution/result index, retaining failures and any work
   left incomplete by the deadline.

The fixed model/schedule definitions are described in [revision-v2.md](revision-v2.md).
Learning rate is 0.0003 with cosine decay to 0.000003 over 60 epochs; λ=1 for the
physics variants. All variants use effective batch 256, microbatch 64, and evaluation
batch 32. Training arrays through t=1 are cached in RAM (approximately 207 MB for
the full training set), preserving exactly the selected arrays and pair ordering.
No model, rate, λ, seed, or subset is chosen based on observed test performance.

## Preflight evidence and budget

All four production-sized variants passed CPU/Metal prediction and gradient checks.
Maximum prediction disagreement was below 6e-8. The regression suite passed all
35 tests after adding the training cache, including exact cached/noncached pair
equality and resumed training.

A five-measured-update timing check estimated **13.94 hours** of full-data training
and scheduled validation. The low-data estimate is roughly 1½ hours. These are short
estimates, not measured full-run durations; final diagnostics, I/O, checkpoints,
and sustained thermal throttling add time. CPU stress generation overlaps training.
The controller enforces the deadline rather than silently running indefinitely.
If a stage is unfinished then, it retains checkpoints and reports it as incomplete;
it does not bypass the completion gate or invent final metrics.

Evidence: [GPU preflight](../artifacts/one-day/preflight/verification/doctor-mps.json),
[short timing check](../artifacts/one-day/preflight/verification/benchmark.json),
[fixed execution plan](../artifacts/one-day/plan.json).

## Inspect progress

- [Overall live status](../artifacts/one-day/status.json): active tasks, heartbeats,
  time remaining, completed training counts, and latest training output.
- [Consolidated result index](../artifacts/one-day/ONE_DAY_RESULTS.md): initial task
  index, finalized with available results when the worker finishes or stops.
- [Low-data progress](../artifacts/one-day/lowdata/execution/status.json).
- GPU training logs: `artifacts/one-day/{lowdata,full}/execution/train.log`.
- CPU generation logs: `artifacts/one-day/execution/shock-*.log`.
- Final reports: `artifacts/one-day/{lowdata,full}/report/report.md` and
  `artifacts/one-day/{lowdata,full}/stress-low-viscosity/report/report.md`.
- Frozen source and configuration hashes: `artifacts/one-day/execution/`.

```bash
.venv/bin/python scripts/one_day.py status
tail -n 15 artifacts/one-day/lowdata/execution/train.log
```

The controller requests idle-sleep prevention. Keep the Mac powered and its lid
open; shutdown and lid sleep can still interrupt local execution. Stop deliberately
with `.venv/bin/python scripts/one_day.py stop`. A restart preserves completed
stages and the original deadline; it does not reset the one-day budget.

PDEBench integration is the only planned dataset work deferred until the files
arrive. These custom-data studies remain internal evidence, not an external
benchmark validation. Original v1 artifacts and frozen source are preserved.
