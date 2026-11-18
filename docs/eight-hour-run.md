# Eight-hour replicated pilot

The full dataset and frozen scientific model/training definitions are unchanged.
This run uses seeds 0, 1, 2 for U-Net, vanilla FNO, and spectral physics FNO with
lambda=1. Existing seed-0 checkpoints resume; six additional runs start from their
paired initializations. No architecture, learning-rate, or lambda search is made.

The controller first brings every run to epoch 10, then every run to epoch 20.
Remaining training time extends the least-trained runs one epoch at a time. It
reserves the last 15 minutes of the eight-hour wall-clock budget for validation
and reporting. Runtime variation may prevent every target from finishing.

The controller uses the frozen production modules via PYTHONPATH. A wrapper around
their existing atomic checkpoint writer captures exact epoch boundaries and
returns control to the scheduler. It changes stopping behavior only, not updates,
data order, initialization, losses, or optimizer state. An integration check
verified exact boundary stopping and resumed training on a separate fixture.

Milestone checkpoints are retained in `artifacts/study/eight-hour/milestones`.
Final comparison uses the largest epoch completed by **all nine runs**, rather
than favoring whichever architecture finished more training. Newer unmatched
checkpoints remain available for later continuation. If the time limit interrupts
a training chunk, its last atomic checkpoint is preserved; up to 100 optimizer
updates since that checkpoint may need replaying.

Automatic health checks detect nonfinite losses/gradients. A failed subprocess
stops the controller and records the error; it is not silently excluded. The
controller enforces a deadline and has an exclusive lock to prevent another
production worker from competing for the same artifacts.

Results use all 250 validation trajectories at N=256 for 100-step autoregressive
rollouts, reporting accuracy and normalized mass drift by seed plus aggregate
accuracy mean/SD. Test data remain locked. Resolution transfer and full energy/
PDE diagnostics are sensible follow-up work after inspecting these results.
This remains an early-training pilot, not a claim of model convergence.

Live files:
- `artifacts/study/eight-hour/status.json`: stage, run, target epoch, heartbeat, deadline.
- `artifacts/study/eight-hour/training.log`: checkpoints and validation progress.
- `artifacts/study/eight-hour/validation.log`: final validation output or errors.
- `artifacts/study/eight-hour/results.md`: matched-epoch report when completed.

The job is a local process; keep the Mac powered with its lid open. It requests
idle-sleep prevention. Machine shutdown or lid sleep can still interrupt it.
