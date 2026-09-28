# scripts/cache_generation/

Builds and re-derives the subnet MAC/config caches checked into
`subnet_caches/`.

## Files

- **`build_canonical_cache.sh`** — the canonical top-level orchestrator:
  runs `generate_cache_main_segment.sh`, then
  `generate_cache_low_end_segment.sh`, then `merge_extended_caches.py`, and
  writes `subnet_caches/extended_range_25M_1500M.csv` (56 subnets,
  25M-596M MACs). This is the single command to reproduce the canonical
  cache from scratch.
- **`generate_cache_main_segment.sh`** — runs the entropy-maximizing GA
  fitness search (`src/feast/nas/generate_subnet_cache.py`) over
  `configs/supernets/4-stage-supernet-cifar100-v2.json` for the 50M-600M MAC
  range (50 subnets, rho0=0.31), writing
  `subnet_caches/8_blocks_supernet_cache_alpha_weights_all_1s_rho_0-31.csv`.
- **`generate_cache_low_end_segment.sh`** — same GA search, 25M-75M MAC
  range (6 subnets, rho0=0.40 — relaxed from 0.31 because the tighter
  ceiling has no feasible architecture at this scale), writing
  `subnet_caches/extended_range_low_10M_50M.csv`.
- **`merge_extended_caches.py`** — concatenates the low-end and main
  segment CSVs, sorts by MACs, warns if any two subnets land within 0.1M
  MACs of each other (near-duplicates from separate GA runs), and prints a
  per-MAC-tier count summary.
- **`recompute_macs_for_config.py`** — takes an existing cache CSV and
  recomputes MACs/params under a *different* supernet config (different
  input resolution/stem stride), keeping the architectures (`d`/`e`/`w_indices`)
  identical. Used to derive
  `subnet_caches/extended_range_tinyimagenet_25M_1500M.csv` from the
  CIFAR-100 cache's architectures at TinyImageNet's 64x64/stem_stride=2
  config — not an independent GA search, just a MAC/param recount.

On-demand single-subnet discovery at a specific deployment MAC target (as
opposed to building the full cache) lives in `scripts/post_hoc_extraction/`.
