# PDEBench Burgers compatibility and pilot

The source is the checksum-verified DaRUS PDEBench v8.0 `1D_Burgers_Sols_Nu0.001.hdf5` (10,000 trajectories, 201 snapshots, 1,024 grid points). It is left unchanged. [Source archive](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2986&version=8.0); SHA-256 and download provenance are in `data/pdebench/provenance.json`.

## Compatibility decisions

- The file contains **202 time coordinates but 201 solution snapshots**. The first 201 coordinates are checked against t=0…2 at 0.01 increments; the unmatched final coordinate is excluded. The [upstream generator](https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py) creates one more time coordinate than solution slots.
- The source is periodic on [0,1], with 1,024 cell centers. Fourier resampling to 256 and 512 uses the same initial physical origin as the source; the 1,024 values are unchanged. These grids are all derived from the same finite-resolution PDEBench data. N=1024 is a source reference, not a separately verified N=8192 solver solution. Conversion is Fourier resampling, not PDEBench's default subsampling; scores therefore cannot automatically be compared with published benchmark tables.
- The current upstream generator config uses `epsilon=0.001`, and the diffusive flux uses `epsilon/pi`; our configured PDE viscosity is **0.001/pi**. The flux is `u²/2` in the current generator. Archive v8's 2023 publication follows the upstream fix, but exact generator commit provenance for this specific file is not independently established. Numerical reference residual floors must be inspected; a soft loss on coarse, shock-containing trajectories can conflict with finite-volume data. [Upstream configuration](https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_gen/data_gen_NLE/BurgersEq/config/multi/1e-3.yaml), [historical convention issue](https://github.com/pdebench/PDEBench/issues/51).
- No solver-verification result is fabricated for this external dataset. Converted shard checksums, finite-value checks, split IDs, source SHA-256, and normalization are recorded. Training normalization uses only the 200 training trajectories at N=256 through t=1. The test data do not contribute.

## Fixed pilot split and protocol

The official data file is one training archive; this pilot uses indices 0–199 for training, 8750–8999 for validation, and every other index 9000–9999 for the 500 test examples. There are no shared trajectory IDs. This split and reduced data budget are documented adaptations, not a claim to reproduce an official PDEBench split or official score. Test trajectories are only used after validation checkpoint selection.

The pilot fixes four existing model variants, two seeds, 60 epochs, same optimizer and schedule as the current custom-data study, and 100/200-step evaluation at N=256/512/1024. Compare paired variants within this same protocol. The two-seed uncertainty is limited. Since PDEBench changes both the physical problem/distribution and numerical source, differences against the custom study cannot be attributed to a single factor.

- Configuration: `configs/pdebench.yaml`.
- Converter/validator: `scripts/prepare_pdebench.py`. Existing generic `validate-data` is for the generated dataset and must not be used to validate this imported data; the converter performs dedicated checks and writes its own `verification/data.json`.
- Test: `tests/test_pdebench.py`, exercises synthetic source import, split isolation, train-only normalization, and training/evaluation.
- Queue: `scripts/pdebench_queue.py` converts on CPU while the current study runs, waits for it to finish, then runs GPU doctor, training, test evaluation, and report. The queue stops at **17 November 2026, 06:00 IST**; completed checkpoints remain.
- Live status: `artifacts/pdebench/queue-status.json`. Logs: `artifacts/pdebench/{conversion,doctor,train,evaluate,summarize,build-report}.log`. The report, when complete, is `artifacts/pdebench/report/report.md`.

The report generator was originally written for the custom study. For this imported dataset, rely on the manifest and this document for source provenance; disregard generic report language implying our solver generated the PDEBench reference. No PDEBench results are claimed until the benchmark run and checks finish.
