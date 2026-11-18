# Research protocol — burgers-v1

The accepted proposal is preserved in `../plan.md`; the original supplied text is
in `proposal-original.md`. Executable values live in `../configs/study.yaml`.
Fixture settings are strictly for integration verification and are labelled in
every generated dataset, evaluation, summary, and report.

## Questions and scope

Does a soft PDE residual penalty improve rollout error, conservation, energy
behavior, stability, or resolution transfer relative to the same FNO trained with
data loss alone? How sensitive is that conclusion to lambda and the spatial
derivative method? A periodic 1D U-Net is the spatial baseline.

No external dataset, coordinate PINN, viscosity shift, hard conservation
projection, rollout-training loss, or higher-resolution physics supervision is
included. This is a controlled PINO-style adaptation, not a reproduction of all
PINO features. Numerical trajectories are reference solutions, not exact truth.

## Frozen experimental choices

The PDE is `u_t + (u²/2)_x = 0.01 u_xx` on the unit torus. Generate 2,000/250/500
independent train/validation/test trajectories, each sampled at 201 times from
0 to 2 at interval 0.01. The reference grid is 8,192. Model grids are 256, 512,
and 1,024. Learning uses only 256 and times 0–1: 200,000 adjacent training pairs.

The 64-mode initial random Fourier series has independent sine and cosine
coefficients with variance proportional to `((2πk)² + 25)^(-2)`. A fixed scale
sets its expected spatial RMS to 0.5; individual realizations are not rescaled.
Add a mean uniformly distributed on [-0.5,0.5]. This is a custom distribution,
not a claim of reproducing the original FNO dataset.

Five seeds (0–4) cover U-Net, FNO, and physics FNO with lambda in 0.1/1/10/100
and spectral/finite-difference derivatives: 50 runs. Paired FNO runs initialize
the same model and use identical epoch permutations. Each run uses Adam,
learning rate 0.001, no weight decay, 500 epochs, and halving every 100 epochs.
The effective batch is 256; the last partial batch retains all remaining pairs.
Microbatch gradients are weighted by their fraction of that effective batch.

Checkpoint selection uses mean time-averaged relative L2 error over all 250
validation trajectories, rolled from t=0 to 1. Validate every ten epochs.
Any nonfinite or divergent trajectory makes its score infinite. Ties retain
the earlier checkpoint. Lambda is selected per derivative method by mean best
validation score across all five seeds; ties choose the smaller lambda.
No seed is discarded. A failed run is recorded and blocks test release pending
investigation rather than being silently replaced with another seed.

## Statistics and failures

Report both 100- and 200-step horizons. Seed SD measures training variability.
Paired effect intervals use 10,000 bootstrap replicates, sampling seed indices
and shared trajectory IDs with replacement. The same indices apply to both
methods. Timesteps are never independent samples. Intervals are descriptive,
not multiplicity-adjusted hypothesis tests.

A trajectory fails upon nonfinite output or predicted RMS exceeding 100 times
initial reference RMS. Failure remains sticky; failed relative errors are
infinite internally and null in JSON, accompanied by failure flags. Aggregate
metrics are unavailable when any trajectory in that group fails. Failure rates
remain reported; there is no survivor-only mean.

## Change log

2026-11-14: Implemented the approved protocol. Added the explicit divergence
threshold (100× initial RMS), relative-norm floor (1e-8× initial RMS, minimum
1e-12), bootstrap count (10,000), and numerical verification seeds. These fill
previously unspecified execution details without reducing scientific scope.
Further scientific changes require a new protocol version and work directory.

Storage-policy correction: the arbitrary 100 GB free-space threshold was removed.
The check now uses array sizes and explicit working allowances. All scientific
settings, dataset contents, resolutions, seeds, and training budgets are unchanged.
