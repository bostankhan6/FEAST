# experiments/07_gamma_ablation/

Sweeps the compute-data correlation parameter gamma (root README component
2) across FEAST and all three baselines on CIFAR-100 mixaug. The gamma=1
canonical anchor lives in `03_cifar100/`, not here; this folder covers the
other gamma values needed for the paper's gamma-sensitivity figure.

## Files

- **`FEAST_cifar100_gamma0.sh`** — FEAST at gamma=0.0 (compute-data
  independent allocation). Byte-identical to
  `03_cifar100/FEAST_cifar100.sh` except `--corr_gamma 0.0` and
  wandb/checkpoint naming.
- **`FEAST_cifar100_gamma025.sh`** — FEAST at gamma=0.25 (weak positive
  coupling).
- **`FEAST_cifar100_gamma05.sh`** — FEAST at gamma=0.5.
- **`FEAST_cifar100_gamma15.sh`** — FEAST at gamma=1.5 (stronger-than-canonical
  positive coupling).
- **`fiarse_cifar100_gamma0_v3.sh`**, **`heterofl_cifar100_gamma0_v3.sh`**,
  **`scalefl_cifar100_gamma0_v3.sh`** — the gamma=0 arm for each baseline
  (the gamma=1 arm reuses the completed `03_cifar100/` v3 runs rather than
  retraining). "v3" = the post RandAugment-forwarding-bugfix rerun, same as
  `03_cifar100/`'s baseline scripts.

Together with `03_cifar100/`'s gamma=1 runs, this folder produces the
gamma=0-vs-gamma=1 motivation panel across all four methods, not just FEAST.
