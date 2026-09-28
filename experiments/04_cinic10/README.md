# experiments/04_cinic10/

Same structure as `03_cifar100/`, run on CINIC-10 instead. CINIC-10 is 32x32
(no stem change from CIFAR-100) but partitioned far more non-IID
(`--partition_alpha 0.005` vs CIFAR-100's `0.1`), and has its own explicit
`valid/` split so no `--validation_split` flag is needed.

## Files

- **`FEAST_cinic10.sh`** — canonical FEAST run on CINIC-10, same mixaug
  protocol and cache (`extended_range_25M_1500M.csv` — shared with
  CIFAR-100 since both are 32x32) as `03_cifar100/FEAST_cifar100.sh`.
- **`run_fiarse_cinic10.sh`**, **`run_heterofl_cinic10.sh`**,
  **`run_scalefl_cinic10.sh`** — the three baselines at matched protocol,
  mirroring `03_cifar100/`'s equivalents with `--dataset cinic10` and the
  sharper partition alpha.
