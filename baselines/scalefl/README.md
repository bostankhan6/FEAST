# baselines/scalefl/

ScaleFL (Ilhan et al., "ScaleFL: Resource-Adaptive Federated Learning with
Heterogeneous Clients", CVPR 2023). Structured **2D (depth x width)**
scaling: each of 4 fixed complexity "levels" gets both fewer blocks (early
exit) and narrower channels (top-left slice), with early exits trained via
self-distillation from the deepest exit.

## Files

- **`split_config.py`** — computes, for each target compute ratio, the most
  "balanced" `(s_d, s_w)` split (depth fraction, width fraction) such that
  `|s_d - s_w|` is minimized subject to hitting the target MACs within
  tolerance (paper Section 3.1.1) — a brute-force search over exit position
  x width-ratio candidates, using an analytic MAC counter
  (`compute_macs()`) for this repo's ResNet (hidden=[64,128,256,512], 2
  blocks/stage, 8 blocks total, CIFAR 32x32 stem). `exit_channels()` returns
  the channel count an early-exit classifier needs at a given split.
  `get_default_configs()` caches the result of `find_split_configs()` for
  the default 4 levels (ratios `0.061, 0.13, 0.25, 1.0`). Runnable directly
  (`python -m baselines.scalefl.split_config`) to print the resolved
  configs for inspection.
- **`model.py`** — `ScaleFLResNet`. Same backbone as `heterofl/model.py`
  (no `Scaler` module here, though — ScaleFL doesn't use one) plus early
  exit classifiers inserted at the block boundaries `split_config.py`
  resolved. A client at level `l` runs blocks `0..n_blocks[l]-1` at width
  `s_w[l]` and returns logits from every exit up to and including `l`
  (`[logits_exit_0, ..., logits_exit_{l-1}]`) — the global model (level 4)
  runs all 8 blocks and all 4 exits. `build_client_model(level)` /
  `build_global_model()` are the convenience constructors.
- **`federation.py`** — `_build_level_meta()` derives per-level block
  counts, width ratios, and MACs from `split_config.py` once at import
  time (`N_BLOCKS_PER_LEVEL`, `S_W_PER_LEVEL`, `LEVEL_MACS`, `FULL_MACS`).
  `budget_to_level()` picks the highest-affordable level for a client's MAC
  budget. `ScaleFLFederation`: distribute/combine follow the same top-left
  channel-slice pattern as HeteroFL, but per-parameter the *minimum level*
  that actually contains that parameter must be tracked
  (`_min_level_for_param()`) since deeper blocks and later exit classifiers
  simply don't exist for low-level clients.
- **`trainer.py`** — standalone training entry point
  (`python -m baselines.scalefl.trainer`). `scalefl_loss()` implements the
  paper's self-distillation objective (Eq. 5): each exit is a weighted
  combination of its own cross-entropy and a KL term against the deepest
  exit (temperature `tau`, weight `beta`), with per-exit weight
  `(i+1)/(l*(l+1))` so deeper (later) exits count more. Same data-loading
  reuse, checkpoint-to-local-dir, and mixup/cutmix support as the other two
  baseline trainers. `evaluate_level()` evaluates the global model at one
  specific level.
- **`run_scalefl.sh`** — canonical CIFAR-100 launch script (100 clients /
  10 per round / 2000 rounds / Zipf(1.2) / γ=1 / lr=0.025 / beta=0.1 /
  tau=3.0).
- **`__init__.py`** — re-exports the public API
  (`ScaleFLResNet`, `build_global_model`, `build_client_model`,
  `ScaleFLFederation`, `budget_to_level`, `LEVEL_MACS`, `S_W_PER_LEVEL`,
  `get_default_configs`). As with `heterofl/__init__.py`, current callers
  (`scripts/evaluate_compare.py`, `scripts/client_param_footprint.py`,
  `tests/test_scalefl.py`) import from the submodules directly rather than
  through this re-export.
- **`__pycache__/`** — compiled bytecode, not source.
