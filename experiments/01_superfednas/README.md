# experiments/01_superfednas/

SuperFedNAS: single-subnet-per-client training with real MaxNet
cosine-annealed aggregation and uniform-random subnet sampling
(`TS_all_random`). Kept as a negative baseline — this is the prior training
approach FEAST replaces, not FEAST itself (see root README). All four
scripts call `train.py` directly (`--model ofaresnet_generic`).

## Files

- **`cifar100_with_original_paper_setting.sh`** — the literal original
  SuperFedNAS paper setting: 20 clients, 8/round, 2000 rounds,
  partition_alpha=100 (near-IID), a single width multiplier (1.0, no elastic
  width search), the original 4-stage supernet definition
  (`max_extra_blocks_per_stage=2`, base channels `[256,512,1024,2048]` —
  different from FEAST's canonical supernet), and the non-canonical
  `subnet_caches/4_stage_cache_60_subnets.csv` cache. No resource
  heterogeneity, no correlated data, no sub-supernet, no per-step min/random/max training.
- **`cifar100_unconstrained_maxnet.sh`** — the real MaxNet aggregation
  mechanism (`--weighted_avg_schedule maxnet_cos_all_subnet`, cosine-annealed
  extra weight for the client that trained the round's designated "largest"
  subnet), under uniform/unconstrained client budgets, using the canonical
  mixaug protocol (4000 rounds).
- **`cifar100_constrained_maxnet_gamma1.sh`** — the same real MaxNet
  mechanism, but under FEAST's full resource-heterogeneous setting (Zipf(1.2)
  client budgets, budget-filtered `TS_all_random_constrained` sampling,
  gamma=1 correlated data). Still missing FEAST's own sub-supernet
  communication and per-step min/random/max training — isolates whether MaxNet
  aggregation alone survives FEAST's harder difficulty regime.
- **`cifar100_constrained_maxnet_gamma0.sh`** — gamma=0 counterpart of the
  above, shortened to 2000 rounds (vs 4000).
