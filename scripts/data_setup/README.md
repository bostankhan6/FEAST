# scripts/data_setup/

Dataset download/preparation scripts, referenced from the root README's
"Data" section. CIFAR-100 and CINIC-10 loaders also download automatically
on first run via `feast.data.*`; these scripts are the standalone/explicit
way to do it up front.

## Files

- **`download_cifar100.sh`** — downloads `cifar-100-python.tar.gz` from the
  official CIFAR site and extracts it to `data/cifar100/`.
- **`download_cinic10.sh`** — downloads and extracts `CINIC-10.tar.gz`
  (Edinburgh DataShare) to `data/cinic10/` (`train/`, `valid/`, `test/`
  ImageFolder-style class subdirectories).
- **`prepare_tinyimagenet.py`** — TinyImageNet has no automated download
  script (the official source requires accepting terms) — download
  `tiny-imagenet-200.zip` manually, unzip it, then run this to restructure
  it from the original layout
  (`train/<wnid>/images/*.JPEG` + flat `val/images/*.JPEG` +
  `val_annotations.txt`) into the canonical flat
  `train/<wnid>/*.JPEG` / `val/<wnid>/*.JPEG` ImageFolder layout this repo's
  data loaders expect. Idempotent — safe to re-run if interrupted.
