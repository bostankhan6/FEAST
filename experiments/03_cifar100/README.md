# experiments/03_cifar100/

FEAST's canonical CIFAR-100 result and the three external baselines at
matched protocol (100 clients, 10/round, 4000 rounds, Zipf(1.2) client
budgets, gamma=1 correlated data, mixaug augmentation package). This is the
paper's headline CIFAR-100 comparison table.

## Files

- **`FEAST_cifar100.sh`** — the canonical FEAST run (`train.py`, `--model
  ofaresnet_generic`): per-step min/random/max training (`--training_strategy feast` +
  `TS_all_random`), sub-supernet communication, Zipf(1.2) resource
  heterogeneity, gamma=1 correlated data (`--weight_dataset_by_budget
  --corr_gamma 1.0`), the canonical `extended_range_25M_1500M.csv` cache,
  mixaug (RandAugment(2,6) + alternating Mixup/CutMix), cosine LR. This is
  the exact command the root README's "Training" section points at.
- **`run_fiarse_cifar100.sh`** — FIARSE baseline (`python -m
  baselines.fiarse.trainer`), same augmentation package for a fair
  comparison; no weight decay / lr=0.05 / lr_global=1.0 per the FIARSE
  paper's Table 3 (kept faithful, not tuned to match FEAST).
- **`run_heterofl_cifar100.sh`** — HeteroFL baseline (`python -m
  baselines.heterofl.trainer`), same augmentation package, wd=1e-4 (raised
  from vanilla to match the other mixaug runs).
- **`run_scalefl_cifar100.sh`** — ScaleFL baseline (`python -m
  baselines.scalefl.trainer`), same augmentation package, wd=1e-4,
  beta=0.1/tau=3.0 (ScaleFL's self-distillation hyperparameters, unchanged
  from vanilla).

All three baseline scripts are suffixed `_v3` internally (wandb project
`hetero_fednas_cifar100_v3`, checkpoint dirs `*_mixaug_v3`).
