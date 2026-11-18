# Revised experimental pipeline (v2)

**Execution update:** the [one-day run](one-day-run.md) now launches both production
training budgets and the stress workflow. Statements below about what had not been
launched describe the implementation handoff, before this execution started.

Implemented on 15 November 2026 in response to the supplied review points.
This is a new experiment: **old data and results are retained, but old model
checkpoints are not resumed into the new architecture/schedule**.

## What changed and what has actually run

| Requested change | Implementation | Evidence / remaining execution |
|---|---|---|
| Schedule appropriate to a short budget | Cosine decay over 60 epochs, initial LR 0.0003, minimum 0.000003 | Schedule and interrupted restart tested; production retraining remains |
| Smaller complete protocol | Four explicitly named variants × three seeds = 12 runs | Entire 12-run integration fixture completed; no bypass of incomplete-run checks |
| Evaluate selected models | Validation every two epochs; lowest validation rollout score selects `best.pt` | All production evaluation uses `best.pt`, never raw epoch-26 milestones |
| Match model capacity | U-Net: 671,457 parameters; FNO: 671,460 real degrees of freedom | FNO width 71, projection 77, otherwise same four layers / 16 modes |
| Residual time stepping | Configurable normalized-state derivative update | Enabled for constrained variant; identity and gradient tests |
| Hard mean conservation | Remove spatial mean of the update | Enabled for constrained variant; tested over 200 steps at N=256/512/1024 |
| Two-step physics | Average residual penalties over both predicted transitions; gradients through both | Enabled for constrained variant; independent formula/gradient test |
| Full diagnostic evaluation | 100/200 steps; three grids; interpolation controls; energy and residuals; paired bootstrap | Complete fixture through report generation; 500-trajectory production test awaits completed new training |
| Low-data ablation | Fixed 200-trajectory subset, shared across variants/seeds; subset-only normalization | Subset membership, normalization, and tamper rejection tested |
| Low-viscosity stress test | Separate ν=0.001 dataset, 300 test-only trajectories; independent solver verification | Tiny ν=0.001 integration case exercised; full 300-trajectory generation/evaluation remains |
| PDEBench | Preserve a clear boundary for future integration | No PDEBench data imported, and no claim of matching its equation conventions |

Verification outputs:

- **35 automated tests passed**, including the legacy pipeline and new regression/integration tests.
- [Production-sized CPU checks](../artifacts/revision-preflight/verification/doctor-cpu.json):
  all four variants passed forward/gradient/optimizer checks. The new variants
  have not yet been verified on MPS or CUDA; the runner performs the selected
  backend's check before training.
- [Core fixture status](../artifacts/revision-fixture-v2/execution/status.json)
- [Core fixture report](../artifacts/revision-fixture-v2/report/report.md)
- [Shift fixture status](../artifacts/revision-fixture-v2/execution/stress-status.json)
- [Shift fixture report](../artifacts/revision-fixture-v2/stress-low-viscosity/report/report.md)

Fixtures are software checks with tiny datasets/models and three epochs, not
scientific evidence. The full 60-epoch studies have **not** been launched as part
of this implementation update. Completing the old 26-epoch pilots does not
satisfy the revised study's gate.

## 1. Fixed comparison and configurations

The main [revision.yaml](../configs/revision.yaml) declares:

| Variant | Backbone | Prediction | Mean projection | Physics penalty |
|---|---|---|---|---|
| `unet` | Periodic U-Net | Direct next state | No | None |
| `fno` | Capacity-matched FNO | Direct next state | No | None |
| `physics-soft` | Same matched FNO | Direct next state | No | One transition, spectral, λ=1 |
| `physics-constrained` | Same matched FNO | Residual derivative update | Yes | Two transitions, spectral, λ=1 |

All FNO variants use paired random initialization and pair ordering for equal
seeds. Their parameter counts match each other. The three-parameter difference
from U-Net is approximately 0.00045%; matching counts does not equate the
architectures' inductive biases or computational cost.

The fourth variant tests a **bundle** of residual stepping, projection, and
two-step loss. It does not isolate those three effects. Keep the ordinary FNO
and soft-physics FNO as direct predictors so their comparison continues to
isolate the soft loss. Additional single-factor variants can be registered in
a separate experiment if causal attribution of each addition is needed.

Configured paired comparisons are soft minus FNO, constrained minus FNO, and
constrained minus soft. Negative error differences favor the first variant.
Both spectral and finite-difference **diagnostic** residuals are measured for
every variant; this core 12-run matrix does not compare models trained with
spectral versus finite-difference losses. The variant schema supports either
derivative method in a separate predeclared ablation.

The original [study.yaml](../configs/study.yaml) remains a 50-run protocol.
`lock_selection` now reads either the legacy matrix or an explicit variant list;
it does not globally reduce the original experiment or treat missing runs as done.

Other configurations:

- [revision-lowdata.yaml](../configs/revision-lowdata.yaml): identical four-variant
  study using 200 of the existing 2,000 training trajectories.
- [shock-evaluation.yaml](../configs/shock-evaluation.yaml): 300 newly generated
  test-only trajectories at ν=0.001; no train/validation split and no fitted
  normalization on those test fields.
- [revision-fixture.yaml](../configs/revision-fixture.yaml): integration fixture.

## 2. Model and objective definitions

For normalized field `z = (u−μ)/σ`, the constrained network calculates

```text
q = Mθ(z)
q_projected = q − mean_x(q)
z_next = z + Δt q_projected
u_next = μ + σ z_next
```

Thus `Mθ` predicts the time derivative in normalized units. The physical update
is `Δt σ q_projected`. Projection conserves spatial mean algebraically; actual
floating-point rollouts can retain roundoff-level drift. It does not enforce
correct shock speed, field accuracy, or energy dissipation.

Implementation: `TimeStepper` and `build_model` in
[models.py](../src/burgers_research/models.py). The wrapper adds no parameters.
Direct prediction remains the default for legacy configurations.

For constrained training, with physical input `u`, true next state `y`, and
predictions `v` then `w`:

```text
L_data = mean(((v−y)/σ)²)
R(a,b) = (b−a)/Δt + [Dₓ(a²/2)+Dₓ(b²/2)]/2
                   − ν[Dₓₓ(a)+Dₓₓ(b)]/2
L_physics = 0.5 × [mean((Δt R(u,v)/σ)²) + mean((Δt R(v,w)/σ)²)]
L_total = L_data + λ L_physics
```

Both predictions are differentiable; the first is **not detached** before the
second model call. Averaging the two residual terms keeps the number of terms
from automatically doubling the effective penalty scale. There is no additional
supervised second-step target. All 200,000 original adjacent training pairs are
retained. For the final pair through t=1, the second prediction reaches t=1.01
using the equation alone; no future reference target is read for that penalty.

Implementation: `training_loss` in [physics.py](../src/burgers_research/physics.py),
called by training, backend checks, and benchmarking. The one-step supervised
and soft-physics paths retain their original definitions.

Cosine decay is stepped once after each epoch, with `T_max=60`. History records
both the learning rate used during the epoch and the next rate. Checkpoints save
and restore scheduler state along with optimizer, sampling cursor, RNG, and
normalization. The configured minimum is reached by the scheduler after the
60th epoch; the final training epoch uses the preceding scheduled rate.

Cosine decay, residual stepping, and pushforward physics are hypotheses to test.
They do **not** guarantee that seed variance disappears or rollouts stabilize.
[PyTorch's scheduler definition](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
specifies the schedule, not an accuracy guarantee.

## 3. Dataset reuse and the low-data experiment

`dataset_protocol: burgers-v1` lets the new experiment reuse the validated
original data without changing its numerical-data identity. Its experiment
signature still includes the new protocol, model specification, training budget,
and evaluation setup. `reuse-data` links the existing data directory and copies
verification metadata into the new work directory. Generation refuses to write
through a linked data directory.

For the low-data study, `train_trajectories: 200` and `subset_seed: 1729` select
training IDs once. The selection is independent of model seed and validation
performance. `training-view.json` records selected IDs, training-only mean/std,
source dataset hash, and its own hash. Every model uses that same view.

Training normalization is recomputed using only selected trajectories at N=256
through t=1. Validation and test evaluation use the resulting checkpoint
normalization. The unselected 1,800 trajectories do not contribute statistics.
There are 20,000 adjacent pairs per epoch instead of 200,000. Validation still
uses 250 trajectories and test evaluation still uses 500, so total wall time
will not necessarily decrease tenfold.

Inspect `training_view`, `reuse_dataset`, and `Store` in
[data.py](../src/burgers_research/data.py).

## 4. Test release, provenance, and metrics

The 12 registered runs must each finish 60 epochs and produce a finite selected
validation score. The gate checks the registered spec, experiment signature,
epoch count, source hash, best-checkpoint hash, common dataset, and training view.
It then freezes the predeclared comparisons and selected checkpoints.
There is no selection of a favorable λ after observing test results: λ=1 is fixed
in this matrix. Changing variants, data budget, or schedule requires a new directory.

The evaluator uses the checkpoint normalization and saved `best.pt`. It writes
`evaluation-context.json` with training/evaluation dataset hashes, configuration,
source identity, and whether evaluation is on a different distribution. Cached
evaluations must match both the context and checkpoint. Summaries check provenance
and reports check saved metric-file hashes.

The existing [metric definitions](../PIPELINE_WALKTHROUGH.md#5-exactly-what-the-reported-metrics-mean)
continue to apply. New field **`extrapolation_error`** averages per-time relative
L2 only after the training window:

```text
mean(e_101, …, e_200)   # t=1.01…2 for the main configuration
```

The rollout still starts at t=0 and is never reset at t=1. This metric is reported
beside the ordinary 200-step average, terminal error, failure rate, conservation,
energy error, dissipation balance, and spectral/finite-difference residuals.
The paired bootstrap also covers the extrapolation metric.

Native N=512/1024 inference is compared with Fourier restriction to N=256 plus
cubic/linear interpolation on every step. Reference, persistence, and oracle
interpolation controls remain. Physics diagnostics use the **evaluation PDE's
viscosity**, including during the ν=0.001 shift experiment.

The production configuration uses 10,000 paired seed-and-trajectory bootstrap
replicates. With only three seeds, intervals are descriptive and limited;
trajectories and timesteps must not be presented as independent training repeats.
Any failed rollout suppresses that group's aggregate metric instead of silently
averaging only successful trajectories.

## 5. Low-viscosity stress experiment

The proposed generated stress set keeps the unit periodic domain and initial
condition family, changes ν from 0.01 to 0.001, and uses independent generation
seed 20261116. It has 300 test trajectories, reference N=8192, projected grids
256/512/1024, and times 0…2. It starts solver verification at internal dt=0.00025;
the solver must pass time/grid refinement checks at the new viscosity before
generation. Numerical divergence triggers timestep refinement during verification;
insufficient spatial resolution still stops generation rather than relaxing tolerance.

This tests a **physical-parameter shift** of models trained at ν=0.01. The models
do not receive viscosity as an input. Failure under this shift does not by itself
prove that an architecture cannot learn ν=0.001 when trained on that regime.

During generation, `verification/stress-data.json` records each trajectory's
peak absolute reference gradient and peak fraction of fluctuating spectral energy
at modes k≥16. This checks whether the data actually exercise steep/high-frequency
behavior; low viscosity alone is not treated as proof that every sample has a
sharp front. Both residual discretizations and reference residual floors are
evaluated so discretization error is visible.

Generated ν=0.001 here means the coefficient of `u_xx` in our explicit equation.
Do not infer PDEBench's effective PDE from a filename or substitute `0.001/π`
without inspecting the supplied file and generator conventions. There is an
[upstream discussion of Burgers coordinate/flux conventions](https://github.com/pdebench/PDEBench/issues/51).
An independent benchmark remains necessary; generating a tougher custom set
does not address the shared-solver/external-validity limitation.

## 6. Running and inspecting the new pipeline

Use a fresh work directory for each protocol. The serial runner freezes source
and config, keeps a per-experiment exclusive worker lock, saves stage logs and
heartbeats, and resumes completed stages/checkpoints. A stage failure stops the
workflow. Run these commands from the project root.

Prepare the full-data experiment without training:

```bash
.venv/bin/python scripts/run_revision.py prepare \
  --config configs/revision.yaml --work-dir artifacts/revision --device mps
```

This reuses `artifacts/study/data`, validates it, records the matrix, and runs
backend checks. Then run the full sequence through the final test report:

```bash
.venv/bin/python scripts/run_revision.py run \
  --config configs/revision.yaml --work-dir artifacts/revision --device mps
```

For the separate low-data study:

```bash
.venv/bin/python scripts/run_revision.py run \
  --config configs/revision-lowdata.yaml --work-dir artifacts/revision-lowdata --device mps
```

Run these studies serially on a single accelerator. The commands run in the
foreground; on macOS, prefix with `caffeinate -i` if idle-sleep prevention is
desired. Stop with Ctrl-C; resume using the same command and frozen configuration.
On a system without Metal, select an available `cpu` or `cuda` backend instead.

After a main study completes, generate and evaluate the stress set:

```bash
.venv/bin/python scripts/run_revision.py stress \
  --config configs/revision.yaml --work-dir artifacts/revision \
  --stress-config configs/shock-evaluation.yaml \
  --stress-data-root artifacts/shock-evaluation --device mps
```

The stress runner checks that model selection is ready **before** generating
the stress dataset. Its outputs are separate from ordinary test outputs under
`artifacts/revision/stress-low-viscosity/`. The same validated stress data can
subsequently be evaluated with the low-data study; do not regenerate concurrently.

Inspect without launching work:

```bash
.venv/bin/python scripts/run_revision.py status --work-dir artifacts/revision
.venv/bin/python scripts/run_revision.py status --work-dir artifacts/revision --stress-status
```

Once generated, inspect:

| Evidence | Location relative to the study directory |
|---|---|
| Frozen code/config and hashes | `execution/source/`, `execution/source/snapshot.json` |
| Stage progress / errors | `execution/status.json`, `execution/<stage>.log` |
| Actual learning rates and validation history | `runs/<variant>-s<seed>/history.json` |
| Selected model / recovery checkpoint | `runs/<variant>-s<seed>/best.pt`, `last.pt` |
| Subset IDs and normalization (low-data only) | `training-view.json` |
| Locked comparisons/checkpoint hashes | `selection.json` |
| Evaluation distribution and provenance | `evaluation-context.json` |
| Every trajectory, metric, and time curve | `evaluation/<variant>-s<seed>/n<N>-<adapter>.json` |
| Summary rows / bootstrap intervals | `summary/metrics.csv`, `summary/summary.json` |
| Report and figures | `report/report.md` |
| Shift report | `stress-low-viscosity/report/report.md` |

To reproduce the small end-to-end software check in another directory:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/run_revision.py run \
  --config configs/revision-fixture.yaml --work-dir artifacts/revision-fixture-repeat \
  --generate-fixture --device cpu
.venv/bin/pytest -q
```

Use the frozen v1 source when inspecting/resuming old v1 checkpoints. Current
source intentionally rejects resuming them under altered code. Existing
`artifacts/study`, epoch-26 analyses, and their frozen source were not modified.

## 7. Interpreting the supplied review

The review's suggested experiments are implemented, but its predicted outcomes
remain unproven. Specifically, we are not assuming that cosine decay eliminates
seed variance, that direct prediction is the sole cause of drift, that soft
physics is redundant, that the smooth problem is saturated, that low data makes
baselines diverge, or that low viscosity guarantees an FNO disadvantage.

The physics-FNO comparison must preserve failures and unfavorable outcomes.
Projection makes conservation partly a design property, so field accuracy,
long-horizon behavior, and energy diagnostics are necessary alongside mass drift.
PDEBench integration and a frozen external-benchmark protocol are the next data
work once the files arrive; none of the new custom-data experiments substitutes
for that comparison.
