# PDEBench — 600 training trajectories, three models, two seeds

**Superseded during training by the [64-mode FNO revision](pdebench-600-fno64.md).**
Use that document's status and report links for the active study.

Protocol: `pdebench-600trajectories-unet-fno-constrained-unet-2seeds`.

| Setting | Value |
|---|---|
| Models | Plain U-Net, plain FNO, constrained U-Net |
| Seeds | 0 and 1; six training runs total |
| Training | 600 trajectories, source indices 0–599 |
| Validation | 250 trajectories, source indices 8750–8999 |
| Test | 500 trajectories, every other index from 9000–9999 |
| Epochs | 60 per model and seed |
| Effective batch / microbatch | 256 / 64 |
| Learning rate and schedule | 0.0003, cosine, final rate 0.000003 |
| Training grid / time | N=256, t≤1 |
| Training pairs / updates | 60,000 pairs per epoch; 14,100 updates per run |
| Evaluation | N=256/512/1024, 100 and 200 steps |
| GPU runtime limit | Six hours from the doctor stage; data conversion and queue wait excluded |

All six models train from scratch. Constrained U-Net uses residual stepping,
zero-mean update projection, and a spectral physics penalty over two predicted
steps with weight 1, matching the existing constrained variants. It has the same
trainable parameter count as plain U-Net. No hyperparameter search is included.

The original downloaded HDF5 archive is checksum verified before conversion.
The first 600 training trajectories contain the pilot's original 200 as a subset.
Normalization is recalculated from these 600 trajectories at N=256 through t=1.
Held-out IDs remain fixed and disjoint from training. Converted data is stored in
`artifacts/pdebench-600-data`; models and reports are stored separately in
`artifacts/pdebench-600-three-models-2seeds`.

Data preparation can proceed while the previous 200-trajectory constrained U-Net
study runs. GPU stages wait until that study's status is complete. If the prior
study stops or fails, the queue stops for inspection instead of sharing the GPU.
The six-hour timer begins after this wait. On timeout, the workflow retains saved
checkpoints and records `deadline_reached`; a runtime extension requires an
explicit decision. Completion means training, evaluation, summary, and report
generation have all succeeded.

The report explicitly labels grid adapters. Both U-Net variants use cubic
adaptation above the training grid in its main table. Pairwise effect tables
compare native execution for both models. All adapter metrics remain available
in the full summary. N=256 is the simplest direct architecture comparison.

This follow-up was selected after examining earlier pilot test results and uses
the same held-out examples; it is not independent confirmation of the broader
research hypothesis. See [PDEBench provenance and limitations](pdebench.md).

## Inspection

- [Configuration](../configs/pdebench-600-three-models-2seeds.yaml)
- [Live status](../artifacts/pdebench-600-three-models-2seeds/execution/status.json)
- [Conversion log](../artifacts/pdebench-600-three-models-2seeds/execution/conversion.log)
- [Training log](../artifacts/pdebench-600-three-models-2seeds/execution/train.log)
- [Report when complete](../artifacts/pdebench-600-three-models-2seeds/report/report.md)
- [Prior constrained U-Net status](../artifacts/pdebench-unet-constrained/execution/status.json)

Start/resume from the project root, with GPU access:

```bash
.venv/bin/python scripts/run_pdebench_variant.py \
  --config configs/pdebench-600-three-models-2seeds.yaml \
  --work-dir artifacts/pdebench-600-three-models-2seeds \
  --source-data-root artifacts/pdebench-600-data \
  --prepare-source data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5 \
  --wait-for-status artifacts/pdebench-unet-constrained/execution/status.json \
  --runtime-hours 6
```
