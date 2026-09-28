# baselines/fiarse/

FIARSE (Wu et al., "FIARSE: Model-Heterogeneous Federated Learning via
Importance-Aware Submodel Extraction", NeurIPS 2024). One dense global
model; per-client sub-models are produced by **unstructured, magnitude-based
parameter masking** rather than structured channel/depth slicing — every
client shares the same architecture, just with a different fraction of
individual weights zeroed out. `model_size` (0, 1] = fraction of parameters
kept.

## Files

- **`model.py`** — `FIARSEResNet` (ResNet-18-style, 32x32 input, ~555M MACs
  at `model_size=1.0`). Custom `MaskedConv2d`/`MaskedLinear`/
  `MaskedBatchNorm2d` layers; BatchNorm is never masked (paper: BN stays
  fully shared/active across all sizes). `Bern` is the TCB-GD
  (Threshold-Controlled Biased Gradient Descent) autograd function: forward
  is a hard `|weight| >= threshold` mask, backward substitutes a biased
  gradient that pushes borderline weights decisively above or below the
  threshold instead of just zeroing their gradient. `generate_mask()` does a
  global TopK over `|weight|` across all masked layers to pick the
  threshold for a requested `model_size`. `budget_to_model_size()` maps a
  MAC budget linearly to a model_size (FLOPs scale ~linearly with kept
  parameter fraction under unstructured sparsity), clipped to `[1/256, 1.0]`.
- **`federation.py`** — `FIARSEFederation`. `distribute(model_size, device)`
  deep-copies the global model and applies the mask for that client's size.
  `combine(delta_list, weights)` aggregates client deltas
  (`delta = params_at_receipt - params_after_local_training`) via **partial
  averaging**: at each parameter position, average only over the clients
  whose delta was actually non-zero there (i.e. whose mask included that
  position), then `θ_global -= lr_global * avg_delta`. This is FIARSE's
  most distinctive/error-prone piece of logic — a small-model client must
  never dilute positions it never touched.
- **`trainer.py`** — standalone training entry point
  (`python -m baselines.fiarse.trainer`). Reuses FEAST's own data-loading
  pipeline (`feast.data.*`) so every method sees identical client partitions
  and the same γ-correlated compute/data coupling. Local training is plain
  SGD with **no momentum, no weight decay** (paper Table 3) — sparsity
  enforcement is entirely the Bern backward's job, not the optimizer's.
  Saves checkpoints locally to `--checkpoint_dir` (`best_full.pt` /
  `checkpoint_latest.pt`), independent of wandb. Evaluation sweeps a fixed
  list of model sizes (`EVAL_MODEL_SIZES`, roughly matched to HeteroFL's
  tier MACs for cross-method comparability) with fresh BN calibration
  (`baselines/bn_calibration.py`) at each size.
- **`run_fiarse.sh`** — canonical CIFAR-100 launch script wrapping
  `trainer.py` with the paper-matched settings (100 clients / 10 per round /
  2000 rounds / Zipf(1.2) / γ=1 / lr=0.05 / lr_global=1.0).
- **`__init__.py`** — empty, package marker only.
- **`__pycache__/`** — compiled bytecode, not source.
