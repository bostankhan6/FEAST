#!/usr/bin/env python3
"""
reproduce_training_compute_proxy.py

Computes the analytic forward+backward convolution-MAC training-compute
proxy used in the paper's Training-Computation Controls section (the
"31.01 PMACs" figure cited throughout experiments/10_matched_training_compute/
and elsewhere), for FEAST's canonical CIFAR-100 run.

Implements the proxy exactly as defined in the supplementary
(Sec. D, "Local training computation"):

    C_i,t^fwd = sum_tau n_i,tau * sum_{a in V_i,tau} MAC(a)
    C_hat_i,t^train = kappa_fb * C_i,t^fwd,      kappa_fb = 3

where V_i,tau is the set of variants (global min / local max / a sampled
affordable intermediate) client i trains that mini-batch, per
Eq. (active_variant_set) in the supplementary:
  - {min}                          if local max == global min (1 variant)
  - {min, local max}               if no intermediate exists (2 variants)
  - {min, local max, E[intermediate]}  otherwise (3 variants, using the
    intermediate's *expected* MAC, averaged uniformly over the client's
    affordable intermediate pool, per the paper's "Expected totals" note)

Since a client's variant set and mini-batch count are fixed for the whole
run (only which *specific* intermediate architecture is sampled varies,
and only its MAC is averaged over), the per-round cost C_hat_i is constant
across every round client i is sampled. The federation total is then this
constant summed once per (round, client) appearance:
  - exact: using the realized client-sampling schedule (--schedule), which
    is fully deterministic (seeded by round index) and therefore gives the
    *exact* total for the canonical run, not an approximation.
  - expected: absent a schedule, using the m/N participation model (each
    client expected to be sampled rounds * (m/N) times) from the paper.

Inputs are the per-client partition CSV (and optional per-round schedule
CSV) produced by scripts/theory_constants/dump_canonical_partition.py --
this script does not re-derive the partition itself, to avoid a second,
possibly-diverging implementation of that RNG-sensitive logic.

Usage:
    # 1. Produce the partition/schedule CSVs (see scripts/theory_constants/README.md):
    python scripts/theory_constants/dump_canonical_partition.py \\
        --out results/cifar100/canonical_partition.csv \\
        --schedule-out results/cifar100/canonical_round_schedule.csv

    # 2. Compute the training-compute proxy from them:
    python scripts/reproduce_training_compute_proxy.py \\
        --partition results/cifar100/canonical_partition.csv \\
        --schedule results/cifar100/canonical_round_schedule.csv
"""
import argparse
import ast
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_cache_macs(path):
    """Ascending-sorted MACs list, matching dump_canonical_partition.py's convention."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(float(r["macs"]))
    rows.sort()
    return rows


def load_partition(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "client": int(r["client"]),
                "n_i": int(r["n_i"]),
                "n_affordable": int(r["n_affordable"]),
                "local_max_cache_idx": int(r["local_max_cache_idx"]),
            })
    return rows


def load_schedule(path):
    counts = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cid = int(r["client"])
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def per_client_costs(partition_rows, cache_macs, kappa_fb):
    """Returns list of dicts with per-client fixed per-round cost (feast and
    single-variant reference), and the number of active variants (1/2/3)."""
    mac_min = cache_macs[0]
    results = []
    for row in partition_rows:
        n_aff = row["n_affordable"]
        local_max_idx = row["local_max_cache_idx"]
        mac_max = cache_macs[local_max_idx]

        if n_aff == 1:
            num_variants = 1
            variant_mac_sum = mac_min
        elif n_aff == 2:
            num_variants = 2
            variant_mac_sum = mac_min + mac_max
        else:
            num_variants = 3
            intermediate = cache_macs[1:local_max_idx]
            mean_intermediate = sum(intermediate) / len(intermediate)
            variant_mac_sum = mac_min + mac_max + mean_intermediate

        c_hat_feast = kappa_fb * row["n_i"] * variant_mac_sum
        c_hat_single = kappa_fb * row["n_i"] * mac_max  # single-variant reference: local max only

        results.append({
            "client": row["client"],
            "n_i": row["n_i"],
            "num_variants": num_variants,
            "c_hat_feast": c_hat_feast,
            "c_hat_single": c_hat_single,
            "per_client_ratio": c_hat_feast / c_hat_single,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", default="results/cifar100/canonical_partition.csv")
    ap.add_argument("--schedule", default=None,
                     help="Optional realized-schedule CSV; if given, computes the EXACT "
                          "federation total instead of the expected (m/N model) total.")
    ap.add_argument("--cache", default="subnet_caches/extended_range_25M_1500M.csv")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--clients-per-round", type=int, default=10)
    ap.add_argument("--n-clients", type=int, default=100)
    ap.add_argument("--kappa-fb", type=float, default=3.0)
    args = ap.parse_args()

    cache_macs = load_cache_macs(REPO / args.cache)
    partition_rows = load_partition(REPO / args.partition)
    assert len(partition_rows) == args.n_clients, (
        f"partition has {len(partition_rows)} clients, expected --n-clients={args.n_clients}"
    )

    costs = per_client_costs(partition_rows, cache_macs, args.kappa_fb)

    variant_counts = {1: 0, 2: 0, 3: 0}
    for c in costs:
        variant_counts[c["num_variants"]] += 1

    if args.schedule:
        sample_counts = load_schedule(REPO / args.schedule)
        mode = "EXACT (realized schedule)"
        total_feast = sum(c["c_hat_feast"] * sample_counts.get(c["client"], 0) for c in costs)
        total_single = sum(c["c_hat_single"] * sample_counts.get(c["client"], 0) for c in costs)
    else:
        mode = "EXPECTED (m/N participation model)"
        expected_samples = args.rounds * args.clients_per_round / args.n_clients
        total_feast = sum(c["c_hat_feast"] for c in costs) * expected_samples
        total_single = sum(c["c_hat_single"] for c in costs) * expected_samples

    ratios = [c["per_client_ratio"] for c in costs]

    print("=" * 70)
    print(f"Training-Compute Proxy ({mode})")
    print("=" * 70)
    print(f"  kappa_fb = {args.kappa_fb}, rounds = {args.rounds}, "
          f"clients/round = {args.clients_per_round}, N = {args.n_clients}")
    print()
    print(f"  Clients using 1 variant (min only)          : {variant_counts[1]}")
    print(f"  Clients using 2 variants (min + max)         : {variant_counts[2]}")
    print(f"  Clients using 3 variants (min + rand + max)  : {variant_counts[3]}")
    print()
    print(f"  FEAST total training-compute proxy           : {total_feast/1e15:.2f} PMACs")
    print(f"  Single-variant-reference proxy (same partition): {total_single/1e15:.2f} PMACs")
    print(f"  FEAST / single-variant ratio (federation total): {total_feast/total_single:.2f}x")
    print(f"  Per-client ratio range                        : "
          f"{min(ratios):.2f}x - {max(ratios):.2f}x")
    print("=" * 70)
    print("\nPaper's reported figures (Table: training_compute_proxy, Sec. E.4):")
    print("  31 one-variant / 6 two-variant / 63 three-variant clients")
    print("  FEAST: 31.01 PMACs; single-variant reference: 19.86 PMACs; ratio 1.56x")
    print("  Per-client ratio range: 1.00x - 2.34x")


if __name__ == "__main__":
    main()
