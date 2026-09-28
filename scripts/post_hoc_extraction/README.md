# scripts/post_hoc_extraction/

Searches a single architecture for a requested MAC budget directly from the
trained supernet, after federation has finished — no fine-tuning or further
federated training, just architecture search plus BN recalibration. This is
the post-hoc extraction capability described in the paper: it lets FEAST
serve MAC budgets that weren't in the pre-built 56-subnet cache.

## Protocol

Uses the same training-free GA fitness objective as cache construction
(`src/feast/nas/feast_fitness_maximizer.py`), with population 512, 512
generations, mutation probability 0.3, and an effectiveness ceiling rho0 of
0.40 below 50M MACs or 0.31 at/above 50M MACs — the same rho0 thresholds
`scripts/cache_generation/generate_cache_low_end_segment.sh` and
`generate_cache_main_segment.sh` use. No validation or test accuracy enters
the search; the resulting architecture only receives BN recalibration before
evaluation.

An architecture found on one dataset's supernet config can be reused as-is
on another dataset that shares the same architecture search space
(CIFAR-100 -> CINIC-10 directly; CIFAR-100 -> TinyImageNet with MACs
recomputed for TinyImageNet's 64x64/stem_stride=2 config via
`scripts/cache_generation/recompute_macs_for_config.py`).

## Files

- **`run_single_subnet_search.sh`** — wraps
  `src/feast/nas/search_single_subnet.py`. Takes the target MAC budget as
  its first argument, auto-selects rho0 (0.40 or 0.31) from that target, and
  writes the found architecture (`d`, `e`, `w_indices`, realized MACs/params,
  fitness/effectiveness/entropy scores, and the GA settings used) to
  `subnet_caches/posthoc/target_<macs>.json`.

  ```bash
  ./scripts/post_hoc_extraction/run_single_subnet_search.sh 30e6
  ./scripts/post_hoc_extraction/run_single_subnet_search.sh 500e6 --seed 0
  ```

  Any extra arguments are forwarded to `search_single_subnet.py` (e.g.
  `--seed`, `--output_json` to override the default output path).

- **`extract_and_evaluate.py`** — given a trained FEAST checkpoint and one or
  more of the JSON files `run_single_subnet_search.sh` produces, activates
  that architecture on the checkpoint's supernet (`set_active_subnet` slices
  the weights for whichever architecture is active — this is the
  extraction, no separate copy step needed), BN-recalibrates on the same
  held-out split `scripts/evaluate.py` uses, and evaluates on the official
  test set. Optionally computes the gap against a reference cached-variant
  curve (linear interpolation at the found architecture's realized MAC
  value) when given `--reference_curve`, a `scripts/evaluate.py`-style CSV.

  ```bash
  # Single architecture:
  python scripts/post_hoc_extraction/extract_and_evaluate.py \
      --checkpoint checkpoints/feast-cifar100-mixaug/best_checkpoint_supernet.pt \
      --dataset cifar100 \
      --arch_json subnet_caches/posthoc/target_30000000.0.json \
      --augmentation mixaug

  # Every found architecture in a directory (a MAC sweep), with the gap
  # against the full 56-subnet cached curve:
  python scripts/post_hoc_extraction/extract_and_evaluate.py \
      --checkpoint checkpoints/feast-cifar100-mixaug/best_checkpoint_supernet.pt \
      --dataset cifar100 \
      --arch_json_dir subnet_caches/posthoc/ \
      --augmentation mixaug \
      --reference_curve results/cifar100/feast_eval_full56.csv \
      --output results/cifar100/posthoc_eval.csv
  ```

## Reproducing a MAC sweep

Run `run_single_subnet_search.sh` once per target budget, e.g. looping over
a set of MAC values spanning 25M-560M, to get one found architecture per
target. Then run `extract_and_evaluate.py --arch_json_dir` once over all the
resulting JSONs to get one accuracy (and, with `--reference_curve`, one gap)
row per target.
