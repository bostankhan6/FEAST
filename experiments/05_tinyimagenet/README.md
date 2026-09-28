# experiments/05_tinyimagenet/

Same structure as `03_cifar100/`/`04_cinic10/`, run on TinyImageNet-200
(64x64 input). `--stem_stride 2` downsamples the 64x64 input to a 32x32
post-stem feature map for all four methods, so the body MACs match the
CIFAR-100/CINIC-10 runs. `--partition_alpha 0.1`, same as CIFAR-100.

## Files

- **`FEAST_tinyimagenet.sh`** — canonical FEAST run on TinyImageNet, using
  the TinyImageNet-specific cache
  (`subnet_caches/extended_range_tinyimagenet_25M_1500M.csv` — MACs
  recomputed for the 64x64/stem_stride=2 config, same architectures as the
  CIFAR-100 cache; see `configs/supernets/README.md`), with the same mixaug
  augmentation package as the CIFAR-100 and CINIC-10 FEAST runs
  (`--classifier_dropout 0.0`, mixaug-only augmentation with no
  Cutout/ColorJitter).
- **`run_fiarse_tinyimagenet.sh`**, **`run_heterofl_tinyimagenet.sh`**,
  **`run_scalefl_tinyimagenet.sh`** — the three baselines at matched
  protocol (`--stem_stride 2`, same mixaug package), mirroring
  `03_cifar100/`'s equivalents with `--dataset tinyimagenet`.
