# PDEBench — constrained U-Net only, 200 trajectories, seeds 0 and 1

This extension trains `unet-constrained-s0` and `unet-constrained-s1` from scratch.
It uses the existing PDEBench pilot's exact training, validation, and test splits,
normalization, optimizer, schedule, 60 epochs, and effective batch size 256.
Only the new constrained U-Net is trained and evaluated in this directory.

The constraint package matches the existing constrained FNO: residual stepping
`u_next = u + dt * network(u)` in normalized coordinates, projection of the update
to zero spatial mean, and a spectral physics penalty over two predicted steps
with weight 1. The second step remains in the gradient graph. The wrapper adds
no trainable parameters. The U-Net channels remain [32, 64, 128, 256].

This is a follow-up chosen after inspecting the original pilot, and uses the same
test split for comparison. It is not an independent confirmation experiment.
Checkpoints are selected by validation rollout score, as in the original pilot.

- [Configuration](../configs/pdebench-unet-constrained.yaml)
- [Live status](../artifacts/pdebench-unet-constrained/execution/status.json)
- [Training log](../artifacts/pdebench-unet-constrained/execution/train.log)
- [Report when complete](../artifacts/pdebench-unet-constrained/report/report.md)
- [Original PDEBench report](../artifacts/pdebench/report/report.md)
- [Imported-data provenance and limitations](pdebench.md)

The runner checks the imported manifest and every converted shard checksum,
attaches the existing data, freezes source/configuration, and executes matrix,
device checks, training, evaluation, summary, and reporting. It retains original
conversion validation and does not manufacture a local solver certificate.
It records stage timestamps, a heartbeat, and checkpoints for recovery.

For direct comparisons use N=256 native results, matched by seed and horizon.
The generic standalone report uses native high-resolution scores for the newly
named `unet-constrained` family; the original report uses cubic-adapted scores
for plain U-Net at N=512/1024. Evaluation files retain all three adapters, so
compare native with native or cubic with cubic at those resolutions. A separate
report with only the new variant has no within-report baseline comparison rows.

Start/resume from the project root with GPU access:

```bash
.venv/bin/python scripts/run_pdebench_variant.py
```

The default work directory is `artifacts/pdebench-unet-constrained`. Its own
exclusive worker lock prevents duplicate writers. The report is generated only
after both seeds, all evaluations, and controls complete.
