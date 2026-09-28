# experiments/09_dirichlet_sensitivity/

Sweeps the label-distribution Dirichlet alpha for FEAST and FIARSE on
CIFAR-100 mixaug, holding gamma=1 and Zipf alpha=1.2 (canonical values)
fixed. The canonical `--partition_alpha 0.1` anchor lives in `03_cifar100/`,
not here. No HeteroFL/ScaleFL scripts in this folder.

## Files

- **`run_feast_cifar100_dirichlet.sh`** — parameterized FEAST run (`$1` =
  Dirichlet alpha). Only `--partition_alpha` and run naming change from the
  canonical protocol.
- **`FEAST_cifar100_dir_005.sh`**, **`FEAST_cifar100_dir_03.sh`**,
  **`FEAST_cifar100_dir_10.sh`** — wrappers fixing FEAST's alpha at 0.05,
  0.3, and 1.0 (harsher-than-canonical, mild, and near-IID respectively).
- **`run_all_feast_cifar100_dirichlet.sh`** — runs all three FEAST settings
  sequentially.
- **`run_fiarse_cifar100_dirichlet.sh`** — the FIARSE equivalent of
  `run_feast_cifar100_dirichlet.sh` (same alpha sweep, FIARSE's own
  lr=0.05/lr_global=1.0/no-weight-decay protocol).
- **`fiarse_cifar100_dir_005.sh`**, **`fiarse_cifar100_dir_03.sh`**,
  **`fiarse_cifar100_dir_10.sh`** — wrappers fixing FIARSE's alpha at 0.05,
  0.3, and 1.0.
