# configs/supernets/

Two JSON configs for the same `GenericOFAResNet` search space, differing
only in input resolution/stem stride (and, misleadingly, `n_classes` — see
below). Neither is read by `train.py` (see `configs/README.md`); both are
read by auxiliary cache-generation/analysis/test scripts via
`json.load()`.

## Files

- **`4-stage-supernet-cifar100-v2.json`** — the canonical config. Used to
  generate the canonical subnet cache
  (`subnet_caches/extended_range_25M_1500M.csv`, via
  `scripts/cache_generation/build_canonical_cache.sh`), for the standalone
  architecture searcher, and for CIFAR-100/CINIC-10 (both 32x32 inputs, so
  they share this config). `scripts/client_param_footprint.py`,
  `scripts/reproduce_sub_supernet_communication_reduction.py`, and
  `tests/test_sub_supernet.py` override `n_classes` to the dataset's real
  class count after loading; `src/feast/nas/generate_subnet_cache.py` (the
  GA cache-generation script) uses the config's `n_classes` as-is. The
  MAC/param contribution of the classifier head is negligible relative to
  the body (tens of thousands vs hundreds of millions of MACs).
- **`4-stage-supernet-tinyimagenet.json`** — identical search space, but
  `initial_input_hw=64` (vs 32) and `stem_stride=2` (vs 1), so the stem
  downsamples TinyImageNet's larger input to the same 32x32 feature map the
  body operates on. `n_classes=200` here is correct as-shipped. Used by
  `scripts/cache_generation/recompute_macs_for_config.py` to **re-derive**
  `subnet_caches/extended_range_tinyimagenet_25M_1500M.csv` from the
  CIFAR-100 cache's architectures (same `d`/`e`/`w_indices` per row,
  recomputed MACs/params at TinyImageNet's resolution) — not an
  independent GA search, just a MAC/param recount under a different input
  shape.

## Shared field reference

| Field | Meaning |
|---|---|
| `num_stages` | Number of residual stages (4) |
| `max_extra_blocks_per_stage` | Max additional blocks beyond the mandatory one per stage (8 -> up to 9 blocks/stage) |
| `original_stem_out_channels` | Stem output channels at width_multiplier=1.0 (64) |
| `original_stage_base_channels` | Per-stage output channels at width_multiplier=1.0 (`[128, 256, 512, 1024]`) |
| `initial_input_hw` / `initial_input_channels` | Input spatial size and channel count |
| `stem_stride` | Stem conv stride (1 for 32x32 CIFAR/CINIC, 2 for 64x64 TinyImageNet) |
| `stage_downsample_factors` | Per-stage spatial stride (`[1, 2, 2, 2]`) |
| `channel_divisible_by` | Rounds scaled channel counts to a multiple of this (8) |
| `n_classes` | Classifier output size |
| `width_multiplier_choices` | Elastic width search space, 0.1-1.0 in steps of 0.1 |
| `expansion_ratio_choices` | Elastic per-block expansion ratio search space |
| `bn_gamma_zero_init` | Whether to zero-init the last BN's gamma per residual block |
| `alpha_weights` | **Not** an architecture parameter — per-stage entropy weighting used only by the GA fitness function (`src/feast/nas/feast_fitness_maximizer.py`) during subnet-cache generation. Explicitly excluded when constructing `GenericOFAResNet` (`generic_server_model.py:321-322`). |
| `beta_depth_penalty` | **Not** an architecture parameter either — penalty coefficient in the same GA fitness score discouraging excessive depth. Same exclusion as `alpha_weights`. |
