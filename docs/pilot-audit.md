# Pilot architecture and validation audit

These diagnostics use the existing epoch-10 checkpoints, all 250 validation
trajectories, N=256, and times 0–1. No training or test-set evaluation was run.
Artifacts are in `artifacts/study/quick-results/audit-*.json`.

## Findings

| Model | Teacher-forced one-step relative L2 | Mean 100-step rollout relative L2 | Mean maximum mass drift / initial RMS |
|---|---:|---:|---:|
| U-Net | 0.000853 | 0.01636 | 0.01049 |
| FNO | 0.01468 | 0.51632 | 0.41324 |
| Physics FNO | 0.000646 | 0.03297 | 0.02295 |
| Persistence | 0.02030 | 0.95975 | 0 |

Teacher-forced inputs are true reference states at every step; autoregressive
inputs are model predictions after the first step. Physics FNO has better mean
one-step accuracy than U-Net but worse rollout accuracy. Vanilla FNO is already
less accurate at one step and accumulates substantial mean-field drift.
About 57% of its aggregated squared rollout error is in the spatial mean mode.
That is an error decomposition, not proof of the cause of training instability.
Persistence conserves mass exactly while predicting dynamics poorly, demonstrating
why conservation alone cannot rank models.

The epoch-10 training-average data losses differ far less than the frozen
checkpoint validation errors. Training averages combine changing parameters
throughout an epoch, so they are not the error of its final checkpoint. Training
losses fluctuate and the scheduler has not yet reduced the initial learning rate
(first reduction is epoch 100). We cannot claim convergence or infer that one
architecture is intrinsically inferior from these checkpoints.

## Architecture audit

Both FNO variants instantiate the same model class: four spectral-plus-pointwise
blocks, 64 hidden channels, 16 retained nonnegative modes, GELU, and a 128-channel
projection head. No correction enforces conservation. The Fourier multiplication
and its gradients pass an independent explicit per-mode matrix-product test.

The neuraloperator authors' reference uses the same spectral formulation and
1/(input_channels*output_channels) initialization scale. Our initializer samples
complex normal values, whereas that reference samples complex uniform values.
Our omitted coordinate channel and chosen activation are also documented
adaptations; this is not an exact reproduction. The reference supports selectable
activation. Do not change initialization selectively to favor an observed result.

U-Net uses a conventional encoder/decoder with concatenated skips adapted to 1D
periodic data, with 671,457 real parameters versus FNO's 549,505. Thus the
comparison matches data and training, not exact capacity. A later capacity-matched
ablation can supplement this comparison; it must not replace an unfavorable
baseline result. There is no identified structural fault warranting a selective
architecture change from this audit. CPU/Metal parity on trained checkpoints is
recorded separately; passing these checks is not an exhaustive correctness proof.

References:
- https://raw.githubusercontent.com/neuraloperator/physics_informed/master/models/fourier1d.py
- https://raw.githubusercontent.com/neuraloperator/physics_informed/master/models/basics.py
- https://arxiv.org/abs/1505.04597

## Proposed eight-hour allocation (not launched)

Keep every dataset trajectory and all architecture definitions. Use three seeds
for each of U-Net, vanilla FNO, and spectral physics FNO with lambda=1: nine runs,
each reaching 20 epochs. Reuse the existing seed-0 checkpoints, and start seeds 1
and 2 from their paired initializations. Compare the same epoch-10 and epoch-20
validation checkpoints for every run; retain the existing optimizer schedule.

Measured first-10-epoch times were 17.43, 26.77, and 22.26 minutes. Additional
training to this target extrapolates to about 332 minutes (5.5 hours). Allow about
an hour for validation diagnostics at N=256/512/1024, 100/200-step rollouts,
interpolation controls, conservation/energy diagnostics, and figures, leaving
about 1.5 hours for variability and checkpoint overhead. This is an estimate,
not a guarantee; sustained runtime can differ. An actual eight-hour runner must
enforce a wall-clock deadline, checkpoint at optimizer boundaries, and report
any unfinished runs rather than silently dropping them.

Use validation data for this pilot; keep test data locked for the later fixed
protocol. Report all seeds and matched completed horizons, without selecting
only successful seeds. Twenty epochs provide a replicated early-training
comparison, not evidence of convergence. No lambda/mode/width search is included.

Alternative: continue just the existing three seed-0 runs to approximately 70
epochs (about 6.6 additional training hours), leaving less evaluation time and
providing no replication. The replicated 20-epoch option is recommended for
testing whether the initial ranking is reproducible.
