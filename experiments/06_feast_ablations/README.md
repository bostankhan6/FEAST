# experiments/06_feast_ablations/

Ablates FEAST's own per-step min/random/max training rule (root README
component 1: global-min / random / local-max subnet per step, with KD) to
isolate which piece actually matters. All four scripts are otherwise
identical to the canonical `03_cifar100/FEAST_cifar100.sh` protocol (Zipf(1.2), gamma=1,
mixaug, sub-supernet routing, BN-cal, 4000 rounds) — only the training
strategy / step schedule changes.

## Files

- **`01_min_only.sh`** — every client trains only the global-min (25M MACs)
  variant (`--training_strategy min_only`, codebase label E0). No local-max
  step, no random step, no KD. Structurally equivalent to standard FedAvg on
  a single fixed-size model.
- **`02_max_only.sh`** — every client trains only its local-max
  (largest-affordable) variant (`--training_strategy local_max_only`,
  codebase label E1). No global-min step, no random step, no KD.
- **`03_minmax_kd.sh`** — global-min + local-max steps only
  (`--disable_random_step`, dropping just the random-intermediate step from
  the full min/random/max rule); min still receives KD from the local-max logits.
- **`04_minmax_nokd.sh`** — same schedule as `03_minmax_kd.sh`
  (`--disable_random_step`) but `--kd_ratio 0.0`, so the min step trains on
  plain cross-entropy only, isolating what the KD step itself contributes.
