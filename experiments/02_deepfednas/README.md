# experiments/02_deepfednas/

DeepFedNAS: identical training approach to SuperFedNAS (single-subnet-per-
client, real MaxNet cosine-annealed aggregation) but with Pareto-path-guided
subnet sampling (`TS_optimal_path`) instead of uniform-random. Also kept as a
negative baseline, not FEAST itself. All four scripts call `train.py`
directly, mirroring `01_superfednas/` file-for-file.

## Files

- **`cifar100_with_original_paper_setting.sh`** — the original-paper-scale
  setting (20 clients, 8/round, 2000 rounds, partition_alpha=100 near-IID,
  original 4-stage supernet with `max_extra_blocks_per_stage=2`, base
  channels `[256,512,1024,2048]`), using `TS_optimal_path` sampling and the
  same non-canonical `subnet_caches/4_stage_cache_60_subnets.csv` as
  `01_superfednas/`'s equivalent script — only the sampler differs between
  the two "original setting" scripts.
- **`cifar100_unconstrained_maxnet.sh`** — real MaxNet aggregation under
  uniform/unconstrained budgets with `TS_optimal_path` sampling, using the
  canonical mixaug protocol.
- **`cifar100_constrained_maxnet_gamma1.sh`** — real MaxNet aggregation under
  FEAST's full resource-heterogeneous setting (Zipf(1.2) budgets,
  budget-filtered `TS_optimal_path` sampling, gamma=1), still without
  sub-supernet communication or per-step min/random/max training.
- **`cifar100_constrained_maxnet_gamma0.sh`** — gamma=0 counterpart,
  shortened to 2000 rounds.
