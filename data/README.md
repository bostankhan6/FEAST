# data/

Downloaded/generated dataset content. Entirely gitignored (`data/cifar100/*`,
`data/cinic10/*`, `data/tinyimagenet/*` in `.gitignore`) — nothing under this
folder except this file is (or should be) tracked by git. `git ls-files
data/` returns empty.

A `README.md` placed directly inside `cifar100/`, `cinic10/`, or
`tinyimagenet/` would be swallowed by the `data/<name>/*` gitignore patterns
above, so this one file covers all three subfolders.

## Subfolders (populated at setup time, not checked in)

- **`cifar100/`** — populated by `scripts/data_setup/download_cifar100.sh`,
  which downloads `cifar-100-python.tar.gz` from the official CIFAR site
  and extracts it here. Loaded by `feast.data.cifar100.data_loader`.
- **`cinic10/`** — populated by `scripts/data_setup/download_cinic10.sh`,
  which downloads and extracts `CINIC-10.tar.gz` (Edinburgh DataShare) into
  `train/`, `valid/`, `test/` ImageFolder-style class subdirectories.
  Loaded by `feast.data.cinic10.data_loader`.
- **`tinyimagenet/`** — populated manually: download
  `tiny-imagenet-200.zip` yourself (no automated script — the official
  source requires accepting terms), unzip it, then run
  `scripts/data_setup/prepare_tinyimagenet.py` to restructure it from the
  original nested `train/<wnid>/images/*.JPEG` + flat `val/images/*.JPEG` +
  `val_annotations.txt` layout into a canonical flat
  `train/<wnid>/*.JPEG` / `val/<wnid>/*.JPEG` ImageFolder layout (idempotent
  — safe to re-run). Loaded by `feast.data.tinyimagenet.data_loader`.

Only CIFAR-100, CINIC-10, and TinyImageNet are used by any experiment in
this repo. `data/cifar-100-python/` and `data/cifar-100-python.tar.gz`
(top-level, in `.gitignore`) are unrelated to `data/cifar100/` — a separate
path from before the `data/<dataset>/` convention.

## Populating

```bash
bash scripts/data_setup/download_cifar100.sh
bash scripts/data_setup/download_cinic10.sh
# TinyImageNet: download tiny-imagenet-200.zip manually, unzip into data/, then:
python scripts/data_setup/prepare_tinyimagenet.py
```
