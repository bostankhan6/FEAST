# scripts/

Everything that isn't `train.py` itself: data setup, subnet-cache generation,
evaluation/reproduction tooling, and standalone centralized-training
baselines. Grouped by role below; see each subfolder's own `README.md` for
its files.

## Files

- **`evaluate.py`** — loads a FEAST/SuperFedNAS/DeepFedNAS checkpoint and
  evaluates specified subnets (or the whole cache) on the official test set.
  See root README's "Evaluation" section for full usage.
- **`evaluate_compare.py`** — evaluates and compares FEAST against
  HeteroFL/ScaleFL/FIARSE at the same subnet indices, one checkpoint per
  method.
- **`evaluate_pop_weighted.py`** — turns per-method `evaluate.py` output CSVs
  into the paper's population-weighted accuracy numbers (clients sampled
  from the canonical Zipf distribution, scored by best-affordable subnet)
  plus head-to-head win/tie/loss counts against a baseline.
- **`client_param_footprint.py`** — cross-method client-side parameter
  footprint at matched compute budgets — the basis for any
  bandwidth/communication-cost comparison.
- **`reproduce_sub_supernet_communication_reduction.py`** — validates the
  routed sub-supernet mechanism under the canonical Zipf(1.2) client-budget
  distribution and reports the population-weighted communication reduction
  vs. broadcasting the full supernet (the paper's cited sub-supernet
  bandwidth-reduction number).
- **`inspect_partition.py`** — quick diagnostic: prints per-client
  sample-count and class-count statistics for a given CIFAR-100 Dirichlet
  partition alpha (min/max/mean/median, a class-count histogram, and the
  top-N clients by sample count). Standalone sanity-check tool, not wired
  into any pipeline.

## Subfolders

- **`data_setup/`** — dataset download/preparation scripts.
- **`cache_generation/`** — builds and re-derives the subnet MAC/config
  caches in `subnet_caches/`.
- **`post_hoc_extraction/`** — searches a single architecture for a
  requested MAC budget directly from a trained supernet, then extracts,
  BN-recalibrates, and evaluates it — without fine-tuning.
- **`theory_constants/`** — reproduces the canonical client partition/round
  schedule and the paper's theoretical-appendix constants derived from them.
- **`__pycache__/`** — compiled bytecode, not source.
