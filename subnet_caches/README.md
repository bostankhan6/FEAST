# subnet_caches/

Pre-computed subnet architecture caches: CSVs of `(d, e, w_indices)` architecture
triples with their MACs/params and GA fitness metadata. These are the "optimal
path" caches consumed by `--subnet_cache_path` — used at training time by the
`TS_optimal_path` sampler and the sub-supernet bounds computation
(`GenericServerOFA.compute_sub_supernet_bounds`), and at search time by
`scripts/post_hoc_extraction/` for post-hoc single-subnet extraction. All were
generated (or re-derived) by `scripts/cache_generation/`; see that folder's
README for how to reproduce each file.

## Files

- **`extended_range_25M_1500M.csv`** (56 subnets, 25M-596M MACs) — the
  canonical CIFAR-100 cache, produced by
  `scripts/cache_generation/build_canonical_cache.sh` (merge of the 25M-75M
  low-end segment and the 50M-600M main segment). This is the cache all
  `experiments/03_cifar100/`, `06_feast_ablations/`, `07_gamma_ablation/`,
  `08_zipf_sensitivity/`, and `09_dirichlet_sensitivity/` FEAST runs use.
  Columns: `macs`, `params`, `fitness_score`, `effectiveness`, `entropy`,
  `d`/`e`/`w_indices` (the architecture), and `job_target_macs`/`job_seed`/
  `job_duration_s` (the GA job that produced that row).
- **`extended_range_tinyimagenet_25M_1500M.csv`** (56 subnets, 27M-601M
  MACs) — the same 56 architectures as `extended_range_25M_1500M.csv`, with
  MACs/params recomputed under TinyImageNet's 64x64 input /
  `stem_stride=2` supernet config via
  `scripts/cache_generation/recompute_macs_for_config.py` (not an
  independent GA search). Adds `macs_original`/`params_original` columns
  holding the source CIFAR-100 cache's values for reference. Used by
  `experiments/05_tinyimagenet/`.
- **`8_blocks_supernet_cache_alpha_weights_all_1s_rho_0-31.csv`** (50
  subnets, 50M-596M MACs) — the main-range segment cache
  (`scripts/cache_generation/generate_cache_main_segment.sh`, rho0=0.31)
  that `build_canonical_cache.sh` merges with the low-end segment to
  produce `extended_range_25M_1500M.csv`. Kept standalone for
  reproducibility of the merge step; not referenced directly by any
  experiment script.
- **`extended_range_low_10M_50M.csv`** (6 subnets, 25M-75M MACs) — the
  low-end segment cache (`scripts/cache_generation/generate_cache_low_end_segment.sh`,
  rho0=0.40, relaxed from 0.31 because the tighter ceiling has no feasible
  architecture at this scale). The other half of the merge that produces
  `extended_range_25M_1500M.csv`; not referenced directly by any experiment
  script.
- **`4_stage_cache_60_subnets.csv`** (60 subnets, 458M-3366M MACs) — a
  non-canonical, legacy cache over the original 4-stage supernet config
  (`configs/supernets/4-stage-supernet-deepfednas.json`,
  `max_extra_blocks_per_stage=2`, base channels `[256,512,1024,2048]`,
  width multipliers down to 0.1x). No script in this repo reproduces it —
  it predates `scripts/cache_generation/` and is kept only so the two
  "original paper setting" scripts below still run. Used only by
  `experiments/01_superfednas/cifar100_with_original_paper_setting.sh` and
  `experiments/02_deepfednas/cifar100_with_original_paper_setting.sh` to
  reproduce the original SuperFedNAS/DeepFedNAS paper-scale setting with
  `TS_optimal_path` sampling.
