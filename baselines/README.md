# baselines/

Faithful reimplementations of the three model-heterogeneous federated
learning methods FEAST is compared against in the paper. Each is a
self-contained package (its own model, federation/aggregation logic, and
standalone trainer) — none of them import from `src/feast/`, and FEAST does
not import from them (the one shared dependency is `utils.py` below, used
only to keep client compute budgets identical across methods).

## Files

- **`__init__.py`** — empty package marker, one-line comment
  ("Faithful baseline implementations for DeepFedNAS comparison.").
- **`utils.py`** — `get_client_budgets()`, the exact same per-client MAC
  budget assignment logic as `train.py`'s `_precompute_client_budgets()`
  (Zipf-distributed ranks -> linear MAC mapping -> jitter -> shuffle). Used
  by every baseline trainer and by `scripts/evaluate_pop_weighted.py` so all
  methods (and FEAST) are scored against the *same* sampled client
  population, not independently redrawn ones.
- **`bn_calibration.py`** — `calibrate_bn_running_stats()`. HeteroFL/ScaleFL
  models are trained with `track_running_stats=False` (each tier/level has
  different channel counts, so there's no single running-stat buffer that
  makes sense during training); at evaluation time this function temporarily
  enables running-stat tracking, resets to canonical init values, and runs a
  calibration pass with cumulative-average momentum so `.eval()` doesn't
  silently fall back to per-batch (order-dependent) statistics. Mirrors what
  FEAST gets from `set_running_statistics()` in `scripts/evaluate.py`, so
  baseline evaluation is put on equal footing.

## Subfolders

- **`fiarse/`** — FIARSE (NeurIPS 2024): unstructured magnitude-based
  parameter masking, single global model, partial-average aggregation.
- **`heterofl/`** — HeteroFL (ICLR 2021): structured per-tier channel
  scaling, five fixed model "tiers" (a-e), slice-and-combine aggregation.
- **`scalefl/`** — ScaleFL (CVPR 2023): structured per-level depth+width
  scaling with intermediate classifiers, KD from the largest level down.
- **`__pycache__/`** — compiled bytecode cache, not source; safe to ignore
  (already covered by `.gitignore`).

See each subfolder's own `README.md` for a per-file breakdown.
