#!/usr/bin/env python
"""
evaluate_pop_weighted.py - Population-weighted accuracy across methods.

Per-subnet accuracy (from evaluate.py / evaluate_compare.py) answers "how
accurate is subnet X". It does not answer the paper's actual headline
question: "how accurate is a client, drawn from the canonical Zipf(alpha)
compute-budget population, running the best subnet it can afford". This
script closes that gap.

Algorithm (matches the canonical protocol used throughout the paper, Supplementary
Sec. E.1, "Baseline adaptations and assignment" -- largest-affordable serving under
the canonical Zipf population):
  1. Draw N client compute budgets from the same Zipf population used for
     training (baselines.utils.get_client_budgets — identical logic/seeding
     to train.py's client budget assignment).
  2. For each method and each client budget, pick the highest-accuracy
     subnet/tier whose MACs <= budget (or the cheapest subnet, if none fit).
  3. Population-weighted accuracy = mean accuracy over the N sampled clients.
  4. Optionally compare every method against a --baseline via per-client
     win/tie/loss counts.

Each --method input CSV must have 'macs_m' and 'accuracy' columns, i.e. the
direct output of scripts/evaluate.py (evaluated over the full subnet cache,
not just min/max) or scripts/evaluate_compare.py.

Usage:
    python scripts/evaluate_pop_weighted.py \
        --num-clients 100 --min-mac 25000000 --max-mac 1500000000 \
        --zipf-alpha 1.2 --seed 0 \
        --method FEAST:results/cifar100/feast_eval.csv \
        --method HeteroFL:results/cifar100/heterofl_eval.csv \
        --method ScaleFL:results/cifar100/scalefl_eval.csv \
        --method FIARSE:results/cifar100/fiarse_eval.csv \
        --baseline FEAST \
        --out results/cifar100/pop_weighted_summary.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from baselines.utils import get_client_budgets


def load_anchors(csv_path):
    """Read an evaluate.py/evaluate_compare.py CSV into a sorted (macs_m, accuracy) list."""
    df = pd.read_csv(csv_path).sort_values("macs_m").reset_index(drop=True)
    return list(zip(df["macs_m"].tolist(), df["accuracy"].tolist()))


def best_affordable(anchors, budget_m):
    """Highest-accuracy subnet whose MACs <= budget; cheapest subnet if none fit."""
    fit = [acc for macs, acc in anchors if macs <= budget_m]
    if fit:
        return max(fit)
    return min(anchors, key=lambda t: t[0])[1]


def parse_method_arg(spec):
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--method must be NAME:PATH, got '{spec}'"
        )
    name, path = spec.split(":", 1)
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", action="append", required=True, type=parse_method_arg,
                         dest="methods", metavar="NAME:PATH",
                         help="Method name and path to its evaluate.py-style CSV. Repeatable.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Method name to compute per-client win/tie/loss counts against. Optional.")
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--min-mac", type=float, default=25_000_000)
    parser.add_argument("--max-mac", type=float, default=1_500_000_000)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=0,
                        help="Must match --init_seed used for the canonical training run.")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV path for the summary table.")
    args = parser.parse_args()

    if args.baseline is not None and args.baseline not in dict(args.methods):
        parser.error(f"--baseline '{args.baseline}' is not among --method names: {[n for n, _ in args.methods]}")

    budgets_m = sorted(
        v / 1e6 for v in get_client_budgets(
            num_clients=args.num_clients,
            max_mac=args.max_mac,
            min_mac=args.min_mac,
            zipf_alpha=args.zipf_alpha,
            seed=args.seed,
        ).values()
    )

    per_client_acc = {}
    for name, path in args.methods:
        anchors = load_anchors(path)
        per_client_acc[name] = [best_affordable(anchors, b) for b in budgets_m]

    summary_rows = []
    print(f"{'Method':<12} {'pop_acc':>9} {'min':>8} {'max':>8}")
    print("-" * 42)
    for name, accs in per_client_acc.items():
        pop_acc = float(np.mean(accs))
        row = {
            "method": name,
            "pop_weighted_accuracy": pop_acc,
            "min_client_accuracy": float(np.min(accs)),
            "max_client_accuracy": float(np.max(accs)),
        }
        print(f"{name:<12} {pop_acc*100:>8.2f}% {np.min(accs)*100:>7.2f}% {np.max(accs)*100:>7.2f}%")
        summary_rows.append(row)

    if args.baseline is not None:
        base_accs = per_client_acc[args.baseline]
        print(f"\nHead-to-head vs baseline '{args.baseline}' ({args.num_clients} clients):")
        print(f"{'Method':<12} {'delta_pp':>9} {'wins':>6} {'ties':>6} {'losses':>7}")
        print("-" * 46)
        for row in summary_rows:
            name = row["method"]
            if name == args.baseline:
                continue
            accs = per_client_acc[name]
            wins = sum(1 for b, a in zip(base_accs, accs) if b > a + 1e-9)
            ties = sum(1 for b, a in zip(base_accs, accs) if abs(b - a) <= 1e-9)
            losses = args.num_clients - wins - ties
            baseline_pop_acc = dict(
                (r["method"], r["pop_weighted_accuracy"]) for r in summary_rows
            )[args.baseline]
            baseline_advantage_pp = (baseline_pop_acc - row["pop_weighted_accuracy"]) * 100
            row[f"{args.baseline}_wins"] = wins
            row[f"{args.baseline}_ties"] = ties
            row[f"{args.baseline}_losses"] = losses
            print(f"{name:<12} {baseline_advantage_pp:>+8.2f} {wins:>6} {ties:>6} {losses:>7}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(args.out, index=False)
        print(f"\nSummary saved to: {args.out}")


if __name__ == "__main__":
    main()
