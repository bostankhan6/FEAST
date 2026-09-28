# scripts/theory_constants/

Reproduces the canonical client partition/round schedule and the
theoretical-appendix constants derived from them (finite-population
coverage probability and routed computation scale).

## Files

- **`dump_canonical_partition.py`** — regenerates the canonical CIFAR-100
  gamma=1 client partition, faithfully replaying `train.py`'s RNG flow
  (seeded client-budget assignment via the same Zipf logic as
  `baselines/utils.get_client_budgets`, then the budget-coupled Dirichlet
  partition). For each client, writes sample count (`n_i`), local steps per
  round (`B_i_batches`), raw and capped budget, and the routed sub-supernet
  envelope (`envelope_id`, `envelope_max_d`, `envelope_max_w_indices`,
  `n_affordable`, `local_max_cache_idx`) to a partition CSV. Also writes a
  per-round client-sampling schedule CSV (one row per (round, client) for
  all `--rounds x --clients-per-round` combinations, with the round's cosine
  learning rate `eta_t`).

  ```bash
  python scripts/theory_constants/dump_canonical_partition.py \
    --out results/cifar100/canonical_partition.csv \
    --schedule-out results/cifar100/canonical_round_schedule.csv
  ```

  Defaults match the canonical protocol: 100 clients, seed=0, Zipf(1.2) over
  `[25M, 1.5G]` MACs capped at 600M, partition alpha=0.1, validation
  split=0.1, batch size 64, 10 clients/round, 4000 rounds, lr=0.025, reading
  the canonical cache at `subnet_caches/extended_range_25M_1500M.csv`.

- **`reproduce_theory_constants.py`** — consumes the partition (and,
  optionally, the schedule) CSV above and computes the paper's
  theoretical-appendix quantities for every routed-envelope profile: the
  exact finite-population coverage probability (pi_u) and the routed
  computation scale (psi_u), both derived combinatorially from the client
  sample counts and cohort size (no simulation). If given the schedule CSV,
  it also validates the schedule against the partition (client sample
  counts/envelope IDs must agree, and `eta_t` must match the canonical
  cosine formula) and audits the exact pi_u/psi_u values against what the
  realized 4000-round schedule actually produced, reporting the maximum
  deviation. Writes a per-profile CSV, a JSON summary (client/sample/step
  counts, envelope count, and min/max pi_u and psi_u across profiles), and a
  `.tex` file defining each summary quantity as a LaTeX macro.

  ```bash
  python scripts/theory_constants/reproduce_theory_constants.py \
    --partition results/cifar100/canonical_partition.csv \
    --schedule results/cifar100/canonical_round_schedule.csv \
    --out-dir results/cifar100/theory_constants
  ```

  `--schedule` is optional (drop it to skip the realized-schedule audit).
  `--batch-size` (default 64), `--cohort-size` (default 10), `--rounds`
  (default 4000), and `--eta0` (default 0.025) must match whatever
  `dump_canonical_partition.py` run produced the input CSVs.

## Full reproduction

```bash
python scripts/theory_constants/dump_canonical_partition.py \
  --out results/cifar100/canonical_partition.csv \
  --schedule-out results/cifar100/canonical_round_schedule.csv

python scripts/theory_constants/reproduce_theory_constants.py \
  --partition results/cifar100/canonical_partition.csv \
  --schedule results/cifar100/canonical_round_schedule.csv \
  --out-dir results/cifar100/theory_constants
```

Both steps are deterministic (seed=0 and the documented defaults), so the
output is byte-identical across runs. `results/` is gitignored — see root
README's "Realized canonical experimental artifacts" section for exactly
which of the five items reviewers might ask for come from these two files.
