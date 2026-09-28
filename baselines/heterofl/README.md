# baselines/heterofl/

HeteroFL (Diao et al., "HeteroFL: Computation and Communication Efficient
Federated Learning for Heterogeneous Clients", ICLR 2021). Five **fixed**
model "tiers" (a-e), each a structurally smaller ResNet built by scaling
every stage's channel width by a fixed rate — clients get a genuinely
smaller network, not a masked copy of a bigger one (contrast with FIARSE).

## Files

- **`model.py`** — `HeteroFLResNet` (ResNet-18-style, ~550M MACs at
  `model_rate=1.0`). `scaled_hs = [ceil(rate * c) for c in [64,128,256,512]]`
  — MACs scale as `rate²` since both channel dimensions shrink together.
  `Scaler` is a learnable-rate module applied after each conv+BN
  (`x / rate`) to keep activation magnitudes comparable across differently
  sized clients during aggregation (paper eq. 3). BatchNorm uses
  `track_running_stats=False` — per-tier batch stats only, since there's no
  shared buffer that makes sense across different channel widths.
  `build_heterofl_model(rate)` / `build_global_model()` are the convenience
  constructors.
- **`federation.py`** — `TIER_RATES` (a=1.0, b=0.5, c=0.25, d=0.125,
  e=0.0625) and `TIER_MACS` (`rate² x ~550M`). `budget_to_tier()` picks the
  highest-affordable tier for a client's MAC budget. `HeteroFLFederation`:
  `distribute(tier_name)` returns a **top-left contiguous slice** of every
  global weight tensor (first K output/input channels), cloned so the
  client's copy is independent of the global tensor, sized for that tier. `combine(client_params_list,
  param_idx_list, weights)` accumulates each client's returned slice back
  into the corresponding region of the global tensors and divides by a
  per-position contribution count, so a region only touched by small-tier
  clients isn't diluted by tiers that never wrote there.
- **`trainer.py`** — standalone training entry point
  (`python -m baselines.heterofl.trainer`), same structure as
  `fiarse/trainer.py`: reuses `feast.data.*` for identical partitions
  across methods, local SGD training with mixup/cutmix support, checkpoints
  saved locally to `--checkpoint_dir` (`best_full.pt` /
  `checkpoint_latest.pt`) independent of wandb. `evaluate_tier()` evaluates
  the global model sliced down to one specific tier.
- **`run_heterofl.sh`** — canonical CIFAR-100 launch script (100 clients /
  10 per round / 2000 rounds / Zipf(1.2) / γ=1 / lr=0.025).
- **`__init__.py`** — re-exports the public API
  (`HeteroFLResNet`, `build_heterofl_model`, `build_global_model`,
  `HeteroFLFederation`, `budget_to_tier`, `TIER_RATES`, `TIER_MACS`) at
  package level. In practice every current caller (`scripts/evaluate_compare.py`,
  `scripts/client_param_footprint.py`, `tests/test_heterofl.py`) imports
  directly from the `model`/`federation` submodules instead, so this
  re-export isn't currently exercised — but it's there if needed.
- **`__pycache__/`** — compiled bytecode, not source.
