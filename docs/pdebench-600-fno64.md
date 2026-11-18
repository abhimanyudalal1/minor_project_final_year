# PDEBench 600 trajectories — FNO revised to 64 modes

The active replacement study is `pdebench-600-fno64-2seeds`. It keeps U-Net,
FNO, and constrained U-Net, seeds 0 and 1, 60 epochs, and the existing
600/250/500 training/validation/test trajectories. Its data signature and
normalization match the already converted 600-trajectory archive.

## Choice from training data

The audit uses only the 600 training trajectories at N=256, t=0…1. It removes
the spatial mean, computes real-FFT energy with doubled interior frequency
weights, and evaluates retained energy in both states and adjacent-time
increments. Test trajectories and model test errors do not select the modes.

| Retained modes, including DC | State energy retained | Increment energy retained |
|---|---:|---:|
| 16 | 95.986% | 36.088% |
| 32 | 98.599% | 62.052% |
| 48 | 99.326% | 78.462% |
| 64 | 99.646% | 88.266% |
| 96 | 99.904% | 96.847% |

64 is the smallest audited candidate whose fifth-percentile snapshot state
energy coverage exceeds 99% (99.097%). It is a practical choice with broader
spectral coverage, not an experimentally established optimum. 96 captures more
increment energy and remains a possible later test. These statistics measure
the data spectrum, not a hard limit on FNO output: local layers and nonlinearities
can also represent frequencies beyond the retained spectral weights.

To retain a comparable parameter budget and runtime, FNO width is reduced from
71 to 36. FNO now has 671,879 real parameters versus U-Net's 671,457 (about 0.06%
apart). FNO remains four layers with projection width 77. This jointly changes
width and modes; results cannot isolate a causal effect of mode count alone.
No long hyperparameter sweep is performed.

The original 16-mode study had begun only plain U-Net seed 0 when stopped.
Its saved U-Net checkpoint is copied with an audited protocol-signature migration
because its model, loss, training settings, dataset, normalization, and source
are unchanged. The old checkpoint remains untouched. No FNO weights transfer
between different mode counts. The original six-hour deadline is retained.

- [New configuration](../configs/pdebench-600-fno64-2seeds.yaml)
- [Training-only spectrum audit](../artifacts/pdebench-600-mode-analysis/spectrum.json)
- [New live status](../artifacts/pdebench-600-fno64-2seeds/execution/status.json)
- [Checkpoint migration audit](../artifacts/pdebench-600-fno64-2seeds/checkpoint-migration.json)
- [Training log](../artifacts/pdebench-600-fno64-2seeds/execution/train.log)
- [Report when complete](../artifacts/pdebench-600-fno64-2seeds/report/report.md)
- [Original 600-trajectory setup](pdebench-600-three-models.md)

The prior `pdebench-600-three-models-2seeds` directory is retained as a stopped,
superseded experiment. Inspect this replacement directory for current progress.
