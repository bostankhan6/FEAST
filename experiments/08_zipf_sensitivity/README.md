# experiments/08_zipf_sensitivity/

Sweeps the client-compute Zipf(alpha) parameter for FEAST on CIFAR-100
mixaug, holding gamma=1 and partition alpha=0.1 (canonical values) fixed.
The canonical `--resource_zipf_alpha 1.2` anchor lives in `03_cifar100/`,
not here.

## Files

- **`run_feast_cifar100_zipf.sh`** — the parameterized run (`$1` = zipf
  alpha). Only `--resource_zipf_alpha` and run naming change from the
  canonical protocol. Deliberately sequential (no `--multi_gpu`).
- **`FEAST_cifar100_zipf_08.sh`**, **`FEAST_cifar100_zipf_10.sh`**,
  **`FEAST_cifar100_zipf_15.sh`** — wrappers calling
  `run_feast_cifar100_zipf.sh` with alpha fixed at 0.8, 1.0, and 1.5
  respectively.
- **`run_all_feast_cifar100_zipf.sh`** — runs all three settings
  sequentially in one job.
