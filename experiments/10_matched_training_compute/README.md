# experiments/10_matched_training_compute/

Extends each baseline's round count so its **total** training MACs match
FEAST's canonical CIFAR-100 run (31.01 PMACs), controlling for the
objection "FEAST just trained longer/on more compute." Baselines only —
FEAST's own matched-compute point is just its canonical run in
`03_cifar100/`; there's no FEAST script in this folder.

Each script is otherwise identical to its `03_cifar100/` counterpart (same
seed, partition, budgets, augmentation package, method-specific
hyperparameters) — the round count is the only thing that changes, and
`--lr_cosine` reads `--comm_round` as its horizon directly, so each extended
run gets a fresh full cosine anneal over its new horizon rather than
resuming past an already-completed 4000-round schedule.

## Files

- **`run_fiarse_cifar100_matched_compute.sh`** — 4139 rounds. FIARSE has two
  possible MAC-matching targets since unstructured masking still executes
  as ordinary dense convolutions in PyTorch: "logical active" MACs (what an
  ideal sparse kernel would execute, 19.12 PMACs, ratio 1.62 -> 6487 rounds)
  vs "executed dense-conv" MACs (what this repo's implementation actually
  runs on hardware, 29.97 PMACs, ratio 1.03 -> 4139 rounds). This script
  targets the executed/dense-conv figure as the honest apples-to-apples
  comparison — pass `--comm_round=6487` for the logical-active target
  instead. FEAST still leads by 12.33pp under largest-affordable population
  accuracy after matching (Supplementary Table matched_compute).
- **`run_heterofl_cifar100_matched_compute.sh`** — 8286 rounds (ratio
  31.01/14.97 = 2.07x). FEAST still leads by 23.30pp after matching (down
  from 26.30pp at HeteroFL's original 4000-round compute), so this run
  exists mainly for completeness.
- **`run_scalefl_cifar100_matched_compute.sh`** — 8150 rounds (ratio
  31.01/15.22 = 2.04x). FEAST still leads by 28.51pp after matching (down
  from 31.99pp at ScaleFL's original 4000-round compute).
