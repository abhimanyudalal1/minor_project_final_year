# Current pipeline: execution, inspection, and metrics

**Active work:** [One-day custom-data execution](docs/one-day-run.md), including
live status links and the fixed deadline. No hyperparameter sweep is running.

**Pipeline revision:** the new 12-run, 60-epoch cosine protocol, matched-capacity
FNOs, constrained variant, low-data subset, and low-viscosity evaluation path are
documented in [Revised pipeline v2](docs/revision-v2.md). The walkthrough below
is the **historical v1 experiment and metric audit**, retained to explain the
existing epoch-26 results. Those results were not rerun with the new models.

Status snapshot: **15 November 2026**. The latest completed controller recorded
completion at **17:12:59 IST** (`2026-11-15T11:42:59Z`). This document describes
the code and saved artifacts inspected on that date; it is not a live dashboard.
All paths below are relative to the project root.

## 1. What we are actually doing

We are comparing a periodic 1D U-Net, a vanilla Fourier Neural Operator (FNO),
and the same FNO with an additional soft PDE-residual loss on **our own generated
1D viscous Burgers dataset**. Each network predicts one next state. Repeatedly
feeding predictions back into the network produces an autoregressive rollout.

The current evidence is an **exploratory validation study on a custom dataset**.
It does not establish broad architecture robustness, convergence, or superiority
on an established benchmark. PDEBench has not been integrated. No 2D experiment
has been implemented or run as part of the current study.

| Item | Actual state | Where to inspect |
|---|---|---|
| Custom dataset | 2,750 trajectories; 173 shards; saved integrity check passed | [Data validation](artifacts/study/verification/data.json), [dataset card](artifacts/study/data/dataset-card.md) |
| Original full protocol | 50 runs, five seeds, 500 epochs each; **unfinished** | [Protocol](docs/protocol.md), [configuration](configs/study.yaml), [registry](artifacts/study/experiment-registry.json) |
| LR 0.001 pilot | Nine runs; all compared at epoch 26 | [Pilot status](artifacts/study/eight-hour/status.json), [results](artifacts/study/eight-hour/results.md) |
| LR 0.0003 pilot | Nine runs; epoch-10 and epoch-26 analyses complete | [Latest status](artifacts/study/learning-rate-0003/analysis-epoch26/status.json), [comparison](artifacts/study/learning-rate-0003/analysis-epoch26/comparison.md) |
| Production test evaluation | Not completed; no `artifacts/study/evaluation/` directory at inspection | [Test-selection gate](src/burgers_research/training.py), function `lock_selection` |
| Full diagnostics/report pipeline | Implemented and exercised on small integration fixtures | [Fixture report](artifacts/verified-fixture/report/report.md); **not research evidence** |
| PDEBench / robustness expansion | Discussed as a correction to the experimental design; not executed | Section 10 below |

“Test data remain locked” means no production model-selection/test evaluation
has been released through the evaluation workflow. Test trajectories have
already been generated and read for numerical/data-integrity checks; they are
not physically inaccessible files.

## 2. Start here to inspect the work

| What you want to inspect | File or directory |
|---|---|
| Latest learning-rate comparison | [comparison.md](artifacts/study/learning-rate-0003/analysis-epoch26/comparison.md) |
| Latest mean and per-seed metrics | [results.md](artifacts/study/learning-rate-0003/analysis-epoch26/results.md), [results.json](artifacts/study/learning-rate-0003/analysis-epoch26/results.json) |
| Individual validation trajectories, e.g. FNO seed 0 | [validation-fno-s0.json](artifacts/study/learning-rate-0003/analysis-epoch26/validation-fno-s0.json) |
| Latest controller progress/completion | [status.json](artifacts/study/learning-rate-0003/analysis-epoch26/status.json) |
| Training output and interruptions | [training.log](artifacts/study/learning-rate-0003/analysis-epoch26/training.log), [controller.log](artifacts/study/learning-rate-0003/analysis-epoch26/controller.log) |
| Exact epoch-26 models used in the comparison | [Milestone checkpoints](artifacts/study/learning-rate-0003/analysis-epoch26/milestones/) |
| Latest candidate checkpoints and epoch histories | [Candidate runs](artifacts/study/learning-rate-0003/runs/) |
| Original-rate checkpoint histories | [Control runs](artifacts/study/runs/) |
| Earlier pilot interpretation and figure | [review.md](artifacts/study/eight-hour/review.md), [validation-review.png](artifacts/study/eight-hour/validation-review.png) |
| Teacher-forced / persistence audit | [Audit explanation](docs/pilot-audit.md), [audit-summary.json](artifacts/study/quick-results/audit-summary.json) |
| Dataset IDs, split membership, random seeds | [splits.json](artifacts/study/data/splits.json) |
| Dataset hashes, normalization, numerical checks | [manifest.json](artifacts/study/data/manifest.json) |
| Frozen production source/configuration | [Source snapshot](artifacts/study/execution/source/), [snapshot hashes](artifacts/study/execution/source/snapshot.json) |

The older [production status](artifacts/study/execution/status.json) belongs to
the original persistent worker. It records a stopped generation stage and is
**not the latest pilot status**. Later dataset and pilot artifacts supersede it
for those stages. Likewise, the README and verification-status document contain
historical progress statements; use the relevant controller's status and result
files for current completion evidence.

The latest controller log retains an earlier `KeyboardInterrupt`; its presence
does not by itself mean the resumed job failed. The final status is `complete`,
and nine validation files and the comparison report are present.

## 3. Step by step: data and numerical reference

### Step 1 — Fix the equation and distribution

The equation is

```text
u_t + ∂x(u²/2) = ν u_xx
ν = 0.01, x ∈ [0,1), periodic boundary conditions.
```

Each initial condition is a 64-mode random Fourier series. Independent sine and
cosine coefficients have variance proportional to `((2πk)² + 25)^(-2)`.
A fixed scale sets expected fluctuation RMS to 0.5; individual samples are not
renormalized to that RMS. A random constant mean from `Uniform[-0.5, 0.5]` is added.
This is a **custom distribution**, not a reproduction of PDEBench or the original
FNO dataset.

Inspect [numerics.py](src/burgers_research/numerics.py), especially
`initial_condition`, and [numerical-method documentation](docs/numerics-and-data.md).
The existing [proposal](docs/proposal-original.md) explicitly requested generated
data. That explains provenance, but does not establish external validity.

### Step 2 — Verify the numerical solver

The solver uses CPU float64, a Fourier pseudospectral spatial discretization,
3/2 padding for the nonlinear product, and ETDRK4 time integration. The reference
grid has 8,192 points; the configured starting internal timestep is 0.001.
The saved verification records the accepted internal timestep.

Verification compares timestep refinement, grids 8,192 versus 16,384 on four
development seeds, mass/energy behavior, and an analytic Cole–Hopf case.
These checks support numerical accuracy on the checked cases; they do not prove
accuracy for every possible initial condition.

Inspect `solve`, `verify_solver`, and `physical_checks` in
[numerics.py](src/burgers_research/numerics.py), and the saved
[solver verification](artifacts/study/verification/solver.json).

### Step 3 — Generate trajectories and preserve split identity

Before generation, whole trajectories receive IDs, seeds, and fixed splits:

| Split | Trajectories | Role |
|---|---:|---|
| Train | 2,000 | Fit weights and normalization |
| Validation | 250 | Monitor rollouts, investigate hyperparameters, select checkpoints |
| Test | 500 | Reserved for final evaluation after choices are fixed |

Every trajectory has 201 snapshots from `t=0` to `t=2`, sampled every 0.01.
The float64 reference is Fourier-projected to 256, 512, and 1,024 points and
stored as float32. All resolutions of a trajectory stay in the same split.
Generation checks physical behavior; validation checks hashes, shapes, IDs,
unique seeds, coordinates, and finite arrays.

Inspect `split_manifest`, `generate`, and `validate` in
[data.py](src/burgers_research/data.py). Each HDF5 shard in
[data/](artifacts/study/data/) contains `ids`, `seeds`, `times`, `reference`,
`x_reference`, and `u_256`/`u_512`/`u_1024` with their coordinates. Arrays use
`[trajectory, time, space]`. JSON sidecars record shard hashes and checks.

Current dataset hash:

```text
0f8e25912c3b85e9e106bc9cc09fe0a911869cec73d3a95d171e85cb69161001
```

### Step 4 — Normalize and construct training pairs

Normalization uses only training trajectories at N=256 and snapshots `t=0…1`:

```text
z = (u − μ) / σ
μ = 0.007545282153234323
σ = 0.43179520260932713
```

The 101 snapshots give 100 adjacent pairs per trajectory, or **200,000 training
pairs per epoch**. Inputs and targets use the same normalization. These are
correlated temporal pairs, not 200,000 independent initial conditions.

The base `Store` reads through a bounded HDF5 cache. The lower-learning-rate
wrapper instead caches all base-grid training trajectories through `t=1` in RAM
to reduce I/O. It retains the full training set. Inspect `Store.pairs` in
[data.py](src/burgers_research/data.py) and `CachedStore` in
[learning_rate_study.py](scripts/learning_rate_study.py).

## 4. Step by step: models and training

### Step 5 — Instantiate the models

All models map `[batch, 1, N]` to `[batch, 1, N]` and predict the next normalized
state directly. There is no enforced mean correction or exact conservation layer.

| Model | Implementation | Size / structure |
|---|---|---|
| U-Net | `UNet1d` | 671,457 real parameters; channels 32/64/128/256; circular convolutions, pooling, skip concatenation, transposed-convolution upsampling |
| FNO | `FNO1d` | 549,505 real degrees of freedom; width 64; four Fourier-plus-pointwise blocks; rFFT modes 0–15; projection width 128 |
| Physics FNO | The same `FNO1d` | Same parameter count and paired initialization; additional training penalty |

Inspect [models.py](src/burgers_research/models.py) and
[architecture details](docs/architectures.md). Complex parameters count as two
real degrees of freedom. These are documented architecture adaptations, not
exact reproductions of a published benchmark's implementation. U-Net and FNO
are not parameter-count matched.

### Step 6 — Optimize a one-step objective

Let `u` be the physical input, `v` the physical prediction, and `z_target` the
normalized next-state target. The objective is

```text
L_data = mean((z_pred − z_target)²)
R(u,v) = (v−u)/Δt + [Dₓ(u²/2)+Dₓ(v²/2)]/2
                     − ν[Dₓₓ(u)+Dₓₓ(v)]/2
L_physics = mean((Δt × R(u,v) / σ)²)
L_total = L_data + λ L_physics
```

The spatial derivative is spectral or second-order periodic finite difference;
the time residual uses the trapezoidal average above. Current physics pilots
use **spectral derivatives and λ=1 only**. Baselines use λ=0 and skip residual
computation. The residual is approximate even on numerical reference data.

Inspect `loss`, `residual`, and `spatial_terms` in
[physics.py](src/burgers_research/physics.py).

Training uses Adam, no weight decay, effective batch 256, and deterministic
epoch permutations paired by seed. Microbatches accumulate weighted gradients
before each optimizer update. The configured schedule halves learning rate
every 100 epochs; it **has not reached its first reduction at epoch 26**.

Inspect `train` in [training.py](src/burgers_research/training.py).
`history.json` stores epoch-average `data_loss`, `physics_loss`, and scheduled
`validation_rollout_error`. The loss average includes changing model weights
throughout the epoch; it is not a frozen endpoint metric.

### Step 7 — Checkpoint and distinguish the full protocol from the pilots

The original [production configuration](artifacts/study/execution/source/configs/production.yaml)
specifies 500 epochs, seeds 0–4, four penalty weights (0.1, 1, 10, 100), and two
derivative methods. Together with two baselines, this is **50 runs**.

The completed pilots used just **three families × seeds 0/1/2 = nine runs** at
each learning rate. The LR 0.0003 study has a separate
[configuration](artifacts/study/learning-rate-0003/config.yaml) and run directory;
its data and verification directories link to the original dataset.

| Checkpoint | Meaning |
|---|---|
| `runs/<run>/last.pt` | Latest atomic recovery state; can be partway through an epoch |
| `runs/<run>/best.pt` | Best scheduled validation score so far; not necessarily epoch 26 |
| `milestones/<run>/epoch-26.pt` | Exact epoch boundary used for the matched endpoint comparison |

Checkpoints contain model/optimizer/scheduler state, random-number state,
epoch/cursor/update count, history, normalization, dataset/source hashes,
configuration, and runtime sessions. Periodic recovery saves occur every 100
optimizer updates, plus epoch-end saves.

[budget_run.py](scripts/budget_run.py) wraps the checkpoint writer, saves a
milestone at the requested epoch boundary, and returns control to the pilot
scheduler. Consequently, reaching a pilot target does **not** produce the
full 500-epoch `result.json`. The wrapper may stop before the final epoch's
`history.json` write; inspect the milestone checkpoint's `state.history` when
the text history lags by an epoch.

The eight-hour pilot compared the largest epoch shared by all nine runs: 26.
Some control runs reached epoch 27, but those later checkpoints are not used
in the matched comparison. The LR 0.0003 study completed epoch 10 and then
extended to epoch 26 using [learning_rate_study.py](scripts/learning_rate_study.py).

## 5. Exactly what the reported metrics mean

Define `RMS(a) = sqrt(mean_x(a²))`. For one trajectory, let `u_t` be the reference,
`û_t` the model rollout, and `s₀ = max(RMS(u₀), 10⁻¹²)`. The rollout starts from
the true initial state and then feeds back predictions without reference resets.

### Metrics already reported by the pilots

| Metric / JSON field | Calculation | Interpretation |
|---|---|---|
| Per-step relative L2, `e_t` | `RMS(û_t−u_t) / max(RMS(u_t), 10⁻⁸ s₀, 10⁻¹²)` | Relative field error at one time; 0.1 corresponds to 10% relative error |
| `rollout_error` per trajectory | `mean(e_1,…,e_100)` | Time-average relative error over t=0.01…1; not just the terminal error |
| Run `rollout_error` | Mean of the 250 trajectory rollout errors | Each validation trajectory has equal weight |
| Family `mean` | Mean of the three seed-level run errors | The number used in the model comparison table |
| `seed_sd` | Sample standard deviation of the three run errors (`ddof=1`) | Training-seed variability; not a confidence interval or trajectory SD |
| `normalized_max_mass_drift` per trajectory | `max_t abs(mean_x(û_t)−mean_x(u₀)) / s₀` | Maximum drift in spatial mean, normalized by initial field RMS |
| Run `normalized_max_mass_drift` | Mean of those 250 per-trajectory maxima | Not the maximum across the entire validation set |

On the unit-length periodic domain, the spatial mean equals the discrete
approximation to total mass. The normalization uses initial **RMS**, not initial
mean, so it remains meaningful for near-zero-mean trajectories.

The pilot metric implementation is `summary` in
[budget_run.py](scripts/budget_run.py), reused by the learning-rate controller.
It evaluates N=256 and 100 steps in batches of 16 on MPS; the summary function
hardcodes that batch size, irrespective of the candidate config's runtime
evaluation batch. Its tensors are float32. The full evaluator below computes
its diagnostics in NumPy float64 after model inference.

The headline numbers are neither normalized training MSE nor initial losses.
For example, `0.07917` means about **7.92% mean relative rollout error** under
the averaging procedure above. Squaring, normalization, horizon, and averaging
make training losses and rollout errors numerically different quantities.

### One-step metrics in the earlier audit

Teacher forcing supplies the true `u_t` at every step, rather than feeding back
the previous prediction. `one_step_relative_l2` averages the relative errors
of these independent next-state predictions. The full evaluator additionally
stores `one_step_mse`, the mean physical-space squared error.

Inspect [audit_validation.py](scripts/audit_validation.py) and the
[saved audit](artifacts/study/quick-results/audit-summary.json). The audit used
earlier epoch-10 seed-0 checkpoints, not the latest nine epoch-26 checkpoints.
Persistence (copying the input) had approximately 0.02030 teacher-forced error
but 0.95975 rollout error. This illustrates why small one-step error does not
establish rollout stability.

### Additional implemented metrics, not completed production results

These are calculated by `diagnostics` in
[evaluation.py](src/burgers_research/evaluation.py). Let
`E_t = 0.5 mean_x(û_t²)` and `D_t = ν mean_x((∂x û_t)²)`.

| Field | Exact meaning for horizon H |
|---|---|
| `terminal_error` | `e_H` |
| `max_mass_drift` | Maximum absolute spatial-mean change through H, without RMS normalization |
| `normalized_mass_drift` | The preceding quantity divided by `s₀`; pilot uses the longer field name above |
| `energy_error` | Mean absolute difference between predicted and reference energy over steps 1…H; not relative energy error |
| `energy_increase_count` | Number of increments exceeding `10⁻⁷ × max(E_0,10⁻²⁴)` |
| `dissipation_balance_error` | Maximum absolute `E_t−E_0 + integral_0^t D(s) ds`, with trapezoidal integration |
| `spectral_residual` | Mean over steps of spatial RMS of the physical residual R, using spectral derivatives |
| `finite_difference_residual` | Same, using finite-difference derivatives |
| `failed` | Whether divergence/nonfinite prediction has occurred by the horizon |

Reported residual RMS is **not** the training physics loss: the training loss
squares the residual and scales it by `Δt/σ`. Full evaluation also saves curves
over time, model inference timings, and provenance hashes.

### Failure handling and uncertainty

A rollout fails if predictions become nonfinite or predicted RMS exceeds
`100 × s₀`. In pilot summaries and full evaluation the failure flag remains
set for later steps. Rollout errors become infinity internally; JSON writing
converts nonfinite values to `null`. Treat nulls as unavailable/failed, not zero.
Pilot JSON does not provide the full failure-rate table; the full evaluator
stores explicit failure flags.

The full summarizer suppresses group metric means if any included rollout
fails and reports failure rate. It implements 10,000 paired bootstrap replicates
over seeds and shared trajectory IDs, with 2.5/97.5 percentile intervals for
physics-minus-vanilla differences. It does not resample individual timesteps as
independent observations. Inspect `bootstrap_difference` and `summarize` in
[reporting.py](src/burgers_research/reporting.py).
**Those bootstrap intervals have not been produced for the current pilot.**

## 6. Current results and their limits

Matched epoch 26, all 250 validation trajectories, N=256, 100-step rollouts:

| Model | LR 0.001 mean ± seed SD | LR 0.0003 mean ± seed SD |
|---|---:|---:|
| U-Net | 0.16142 ± 0.20149 | 0.07917 ± 0.04851 |
| FNO | 0.35447 ± 0.31331 | 0.19715 ± 0.22748 |
| Physics FNO, spectral λ=1 | 0.08925 ± 0.11559 | 0.16079 ± 0.20475 |

Source: [saved comparison](artifacts/study/learning-rate-0003/analysis-epoch26/comparison.md).
These are **endpoint checkpoints**, not each run's best validation checkpoint.
The lower learning rate improves mean U-Net/FNO error here but worsens mean
physics-FNO error. Seed spread is substantial; this does not establish a stable
architecture ranking.

The control used microbatch 32. The candidate used 64 for most of the first ten
epochs, 128 for part of seed 0, and 64 thereafter. Effective batch and example
order are retained, but floating-point reduction order can differ. This is an
exploratory comparison with that additional implementation difference.
The validation set also informed the decision to investigate learning rate;
these are not untouched final-test estimates.

## 7. Full pipeline stages and what remains unexecuted

The original runner is [production.py](scripts/production.py); command dispatch
is [cli.py](src/burgers_research/cli.py). Its sequence is:

```text
doctor → matrix → verify-solver → generate-data → validate-data
       → train → evaluate → summarize → build-report
```

`doctor` checks backend/model/gradient behavior. `matrix` registers experiments.
The dataset and training stages are described above. In the full workflow:

1. Scheduled validation runs every ten epochs and selects `best.pt` by lowest
   mean validation rollout error. Equal scores retain the earlier checkpoint.
2. `lock_selection` requires completed results for the entire registered matrix
   and selects λ per derivative method using mean best validation scores across
   five seeds. The selection is saved before the evaluator opens the test split.
3. Test evaluation measures 100/200-step rollouts and teacher forcing at
   N=256/512/1024. At higher resolutions it evaluates native inference and
   base-grid restriction followed by cubic/linear interpolation at each step.
4. Reference, persistence, and oracle-interpolation controls separate learned
   errors from discretization/interpolation effects. Oracle interpolation uses
   the true coarse field at every time and is not an autoregressive model.
5. Summaries, paired intervals, CSV tables, and figures are produced from saved
   evaluation records, with provenance checks.

These production test/report stages remain outstanding. The small
[verified-fixture](artifacts/verified-fixture/) and
[runner-fixture](artifacts/runner-fixture/) outputs exercise the implementation;
they cannot substitute for the research experiment.

## 8. Read-only inspection commands

Run from the project root. These commands inspect saved artifacts without
starting training or opening model evaluation on the test split.

```bash
cat artifacts/study/learning-rate-0003/analysis-epoch26/status.json
cat artifacts/study/learning-rate-0003/analysis-epoch26/comparison.md
tail -n 30 artifacts/study/learning-rate-0003/analysis-epoch26/training.log
cat artifacts/study/verification/data.json
```

Read a specific trajectory record:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('artifacts/study/learning-rate-0003/analysis-epoch26/validation-fno-s0.json')
d = json.loads(p.read_text())
print('Summary:', d['summary'])
print('Trajectory count:', len(d['records']))
print('First trajectory:', d['records'][0])
PY
```

Inspect a trusted, locally generated milestone checkpoint:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import torch
p = Path('artifacts/study/learning-rate-0003/analysis-epoch26/milestones/fno-s0/epoch-26.pt')
c = torch.load(p, map_location='cpu', weights_only=False)
print('Run:', c['spec'])
print('Epoch / cursor / updates:', c['state']['epoch'], c['state']['cursor'], c['state']['updates'])
print('Dataset hash:', c['dataset_hash'])
print('Source hash:', c['source_hash'])
print('Last history rows:', c['state']['history'][-3:])
print('Runtime sessions:', [s['runtime'] for s in c['sessions']])
PY
```

Checkpoint loading here uses pickle-backed state; apply this command only to
the project's trusted checkpoints. For an HDF5 schema inspection without
loading a trajectory array:

```bash
.venv/bin/python - <<'PY'
import h5py
with h5py.File('artifacts/study/data/train-0000.h5', 'r') as f:
    for name, value in f.items():
        print(name, value.shape, value.dtype)
PY
```

The CLI `evaluate` command is a **test-evaluation action**, not a status command.
The pilot controllers' `controller`, `train`, and `summary` modes also perform
work or write artifacts; they are not needed for the inspections above.

## 9. Source provenance and audit limitations

Production execution freezes source under
[execution/source](artifacts/study/execution/source/) and uses that source via
`PYTHONPATH`. At this documentation audit, every current
`src/burgers_research/*.py` file matched its frozen counterpart byte for byte.
The pilot wrapper scripts live under [scripts/](scripts/) and are not included
in the original source snapshot's file list. Their current contents should not
be treated as an immutable record of every earlier launch; checkpoint runtime
sessions, saved configurations, milestone states, and logs provide additional
evidence.

Saved verification documentation records 26 passing automated tests, solver
verification, CPU/MPS comparisons, and completed integration fixtures. Tests
were not rerun merely to create this document. Inspect
[verification-status.md](docs/verification-status.md), [tests/](tests/), and
[verification artifacts](artifacts/study/verification/).

The checkpoint/source checks and selection gate provide workflow protections,
not an access-control boundary. The pilot summary is a separate code path and
does not perform every provenance check in the full evaluator. In particular,
the learning-rate comparison checks the dataset hash and matched epoch, while
the summary loads milestone weights directly. Retain the full artifact trail.

## 10. What needs to change for the intended research claim

The custom generator supports controlled numerical experiments. It does not by
itself establish generalization across datasets, viscosities, initial-condition
families, solvers, or PDEs. Multiple training seeds measure training variability;
they do not establish distribution-shift robustness. The spectral generator and
spectral penalty also share numerical assumptions, making independent benchmark
evaluation useful.

The proposed next protocol should:

1. Integrate **PDEBench Burgers** as an established primary benchmark, pinning
   dataset version/files and verifying equation coefficients, coordinates,
   timestep conventions, splits, preprocessing, and evaluation definitions.
2. Reproduce a documented baseline setting before comparing our modifications.
   Using PDEBench files alone does not make scores comparable if the prediction
   task, horizons, data budget, or metrics differ.
3. Define separate robustness experiments: longer rollouts, resolution changes,
   controlled input perturbations, and held-out physical regimes. A viscosity
   experiment must specify whether viscosity is an input or the test is transfer
   to an unseen equation parameter.
4. Retain this custom dataset as a supplementary controlled study, preserving
   unfavorable results and labeling all exploratory validation choices.
5. Use a new protocol/configuration/output directory for the revised study.
   A 2D experiment is a further extension, not a replacement for benchmark validation.

These changes are **proposed, not implemented in the current pipeline**.
PDEBench provides simulation datasets and reference model implementations;
its relevance is shared benchmarking and external comparison, not that its
data are experimental measurements.
Sources: [PDEBench paper](https://arxiv.org/abs/2210.07182),
[official repository](https://github.com/pdebench/PDEBench),
[dataset archive](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2986).
