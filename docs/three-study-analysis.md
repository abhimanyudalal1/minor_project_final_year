# Analysis of the three completed Burgers studies

Analysis date: 17 November 2026. Based on saved reports, summaries, per-seed
evaluation records, run results, and metric implementation. No new training was
launched for this analysis.

## Evidence and comparison scope

- [Custom 2,000 trajectories, seeds 0 and 1](../artifacts/eight-run/full/report/report.md)
- [Custom 200 trajectories, seed 0](../artifacts/custom-200trajectories-seed0/report/report.md)
- [PDEBench 200 trajectories, seeds 0 and 1](../artifacts/pdebench/report/report.md)

All three use 60 epochs, effective batch 256, four variants, training through
t=1 at N=256, and 500 test trajectories. The custom studies use the same test
trajectories and viscosity 0.01. PDEBench uses different trajectories and a
configured viscosity of 0.001/pi. Its initial-condition distribution and numerical
source also differ. These are separately trained studies, not a test of a custom
checkpoint directly transferred to PDEBench.

The models have approximately 671,460 real parameters each. Comparable parameter
counts do not imply equal optimization difficulty or equal compute: the constrained
variant evaluates a second predicted step in its physics loss.

## Accuracy

Mean rollout relative L2 at N=256, averaged over timesteps 1–100 and test
trajectories. Values are dimensionless; lower is better. The two original studies
average two seeds, while custom-200 has only seed 0.

| Model | Custom 2,000, two seeds | Custom 200, seed 0 | PDEBench 200, two seeds |
|---|---:|---:|---:|
| U-Net | 0.00307808 | 0.02321707 | 0.14207152 |
| FNO | 0.00330765 | 0.03244945 | 0.27232642 |
| Physics-soft FNO | 0.00380756 | 0.03249395 | 0.35230659 |
| Physics-constrained FNO | 0.00077196 | 0.00323117 | 0.22136138 |

The constrained variant wins on both custom studies. Relative to the baseline
FNO, its mean error is about 77% lower with 2,000 training trajectories and 90%
lower in the 200-trajectory run. On PDEBench it improves on FNO by about 19%, but
its mean error is about 56% higher than U-Net's. U-Net is the best tested variant
on PDEBench under this protocol.

For the data-reduction comparison, use seed 0 on both custom studies:

| Model | Custom 2,000, seed 0 | Custom 200, seed 0 | Error multiplier |
|---|---:|---:|---:|
| U-Net | 0.00364342 | 0.02321707 | 6.37x |
| FNO | 0.00297922 | 0.03244945 | 10.89x |
| Physics-soft | 0.00393751 | 0.03249395 | 8.25x |
| Physics-constrained | 0.00080782 | 0.00323117 | 4.00x |

All models deteriorate with the reduced training budget. The constrained variant
has the smallest relative deterioration in this seed. This is promising evidence
for learning with limited data under the chosen schedule, but not an isolated
measurement of sample efficiency: saved run results show 46,920 optimizer updates
per original run versus 4,740 per reduced run. Data diversity, training updates,
and train-derived normalization all change. The single subset and single model
seed do not establish repeatability.

Matching the number of training trajectories does not remove the large custom
versus PDEBench gap. For example, seed-0 FNO error is 0.03245 on custom-200 versus
0.26842 on PDEBench-200. The difference therefore cannot be explained by training
count alone. Attribution to shocks, viscosity, discretization, or initial-condition
distribution requires controlled experiments.

## Long rollouts and failures

At 200 steps, the custom-200 errors are 0.03512 (U-Net), 0.05043 (FNO), 0.05030
(physics-soft), and 0.004261 (constrained). No failures were reported for either
custom study. The constrained advantage persists into the interval after t=1.
The 200-step rollout metric averages the entire t=0.01…2 interval; the separate
`extrapolation_error` metric averages only t>1.

On PDEBench, U-Net/FNO/physics-soft have 200-step errors of 0.19750/0.37006/0.47492.
The constrained aggregate is unavailable because seed 0 has two failed test
trajectories at N=256. This is 2/500 for seed 0, or 2/1,000 seed–trajectory cases
pooled over two seeds (0.2%). Seed 1 has no failures and a mean error of 0.21567.
The failed seed must remain part of the result.

Inspecting the existing seed-0 records identifies `test-09012` and `test-09652`.
Their first null relative-error entries occur at steps 170 and 116 respectively.
The implementation marks a rollout failed when it becomes nonfinite or its RMS
exceeds 100 times its initial RMS; null error alone does not distinguish these
causes. The second trajectory already has 100-step mean relative error 5.8043,
energy error 267.3247, and spectral residual 129,828.84. It is a severe outlier
before the formal failure flag. This shows why a low failure count or a single
average is insufficient to establish stability.

## What the physics metrics establish

Mean conservation is enforced by the constrained architecture. Its roughly
1e-7-scale normalized mass drift is expected from that design. It demonstrates
conservation, but does not demonstrate accurate or stable dynamics. The PDEBench
failures coexist with excellent mean conservation before divergence.

The soft penalty does not consistently improve prediction accuracy. On custom
2,000 it reduces the 100-step spectral residual from 0.02045 to 0.01377 while
increasing rollout error from 0.00331 to 0.00381. With custom-200 its accuracy is
almost identical to FNO. On PDEBench it increases mean error from 0.27233 to
0.35231 and does not improve the mean rollout spectral residual.

The numerical reference itself has a nonzero residual under our diagnostic:
at N=256 and 100 steps, spectral residual is 0.01051 on custom data and 3.38834
on PDEBench. These values are dimensional and not normalized across datasets.
They warrant checking whether the discrete loss is compatible with the imported
reference; they do not alone prove a conversion bug or explain the soft model's
performance. A smaller residual does not guarantee smaller solution error.

The constrained model combines residual stepping, exact mean projection, and
a two-step physics penalty. The current comparison cannot identify the individual
contribution of any one component.

## Resolution and reporting qualifications

The headline high-resolution U-Net rows use a cubic interpolation adapter. FNO
headline rows use native execution. This distinction is absent from the headline
table's columns and must be explicit in a paper. Native U-Net extrapolation error
on custom-2,000 is approximately 0.868 at N=512 and 1.492 at N=1024, versus 0.00694
at its training grid. Cubic adaptation largely restores the training-grid score.
The FNO variants show much more consistent native grid transfer in these data.

However, custom oracle cubic interpolation errors at high resolution are only
about 3e-7 for the 100-step metric. The custom resolution task has very little
unresolved information under this control; nearly identical scores across grids
do not establish performance on more demanding spatial scales. PDEBench's oracle
cubic error is about 0.01, making that distinction more material there.

Two seeds give limited estimates of training variability; one seed provides none.
Bootstrap intervals for custom-200 reflect trajectory variability conditional on
that trained seed. Its `FAILED / unavailable` seed-SD cells mean unavailable SD,
not failed training. PDEBench's unavailable constrained 200-step mean instead
reflects actual failed rollouts. These two cases need different presentation.

PDEBench is a documented pilot using a custom split and Fourier grid conversion,
not a reproduction of an official benchmark score. See [the compatibility
record](pdebench.md). All configurations were fixed for these runs; there is no
completed penalty-strength or residual-discretization sweep despite the generic
report research question mentioning sensitivity.

## Further testing, in priority order

1. **Diagnose stability and residual compatibility first.** Reproduce the two
   failing PDEBench trajectories and inspect state amplitude, energy, spectra,
   and first failure step. Check the same checkpoints on a second backend for
   these cases. Inspect reference residuals and their normalization before
   changing the physics loss. Use validation trajectories for design decisions;
   after adapting methods to observed test failures, reserve a fresh untouched
   confirmation set.
2. **Run component ablations on a fixed 200-trajectory split.** Starting from
   existing FNO and the full constrained variant, add residual-only FNO with no
   physics penalty; residual-plus-mean-projection with no penalty; and that same
   architecture with a one-step penalty. This creates a controlled chain leading
   to the existing two-step variant and helps attribute the gain. Start with one
   seed to diagnose mechanisms, then repeat the relevant comparisons. A physics
   or constrained U-Net becomes important if the claim is architecture-general.
3. **Replicate the reduced-data comparison.** Add custom-200 seed 1 to match the
   existing studies. For stronger claims, use at least three model seeds on the
   principal comparisons and repeat subset selection if claiming sample
   efficiency. Additional seeds assess variability; they do not fix model
   instability by themselves.
4. **Separate data from optimization budget if claiming sample efficiency.**
   Compare selected variants under a predeclared equal-update or equal-compute
   protocol, with explicit scheduling. Keep the present equal-epoch result as
   its own useful question. Increasing PDEBench training count alone would not
   identify which constraint component works or why instability occurs.

There is enough evidence for a preliminary finding: the combined constrained
variant is effective on the custom distribution, including the reduced-data
setting, but its superiority and stability do not extend across the tested
PDEBench conditions. Further targeted testing is needed for a paper claiming
robustness or explaining the mechanism. A broad hyperparameter sweep or a new
2D problem is not necessary to resolve the current evidence gaps.
