#!/usr/bin/env python3
"""Regenerate the canonical CIFAR-100 gamma=1 client partition and dump per-client
sample counts n_i, budgets, and routed-envelope (local-max subnet MAC).

Faithful to the training run's RNG flow:
  - init_seed=0 seeds numpy; _precompute_client_budgets consumes that stream
    (zipf choice + uniform jitter + shuffle) -> budgets.
  - partition_data() internally reseeds numpy to 42 for the validation split and
    then runs the budget-coupled Dirichlet loop, so the partition depends only on
    (budgets, val_split, data), independent of RNG consumed in between.

Canonical config (experiments/03_cifar100/FEAST_cifar100.sh):
  100 clients, Zipf(1.2) over [25M, 1.5G], cap 600M, corr_gamma=1, partition
  alpha=0.1, validation_split=0.1, batch_size 64.

Outputs results/cifar100/canonical_partition.csv and prints B_i summary.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from feast.data.cifar100.data_loader import partition_data  # noqa


def precompute_budgets(n_clients, min_mac, max_mac, zipf_alpha):
    """Exact copy of train.py:_precompute_client_budgets (zipf branch), seeded
    upstream so the RNG stream matches the training run."""
    ranks = np.arange(1, n_clients + 1)
    probs = 1.0 / np.power(ranks, zipf_alpha)
    probs /= probs.sum()
    level_indices = np.random.choice(ranks, size=n_clients, p=probs)
    steps = (max_mac - min_mac) / max(1, (n_clients - 1))
    budgets = min_mac + (level_indices - 1) * steps
    budgets += np.random.uniform(-steps / 2, steps / 2, size=n_clients)
    budgets = np.clip(budgets, min_mac, max_mac)
    np.random.shuffle(budgets)
    return {i: float(budgets[i]) for i in range(n_clients)}


def load_cache(path):
    """Return cache rows sorted ascending by MACs, each with macs, d, w_indices."""
    import ast
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "macs": float(r["macs"]),
                "d": ast.literal_eval(r["d"]),
                "w_indices": ast.literal_eval(r["w_indices"]),
            })
    rows.sort(key=lambda x: x["macs"])
    return rows


def routed_envelope(afford_rows):
    """Per-position bounds over the affordable set = the routed sub-supernet
    envelope (compute_sub_supernet_bounds equivalent): max_d over 4 stages,
    max_w_indices over 5 positions (stem + 4 stages)."""
    nd = len(afford_rows[0]["d"])
    nw = len(afford_rows[0]["w_indices"])
    max_d = [max(r["d"][s] for r in afford_rows) for s in range(nd)]
    max_w = [max(r["w_indices"][p] for r in afford_rows) for p in range(nw)]
    return max_d, max_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-mac", type=float, default=25e6)
    ap.add_argument("--max-mac", type=float, default=1.5e9)
    ap.add_argument("--cap-mac", type=float, default=600e6)
    ap.add_argument("--zipf-alpha", type=float, default=1.2)
    ap.add_argument("--partition-alpha", type=float, default=0.1)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clients-per-round", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--cache", default="subnet_caches/extended_range_25M_1500M.csv")
    ap.add_argument("--data-dir", default="./data/cifar100")
    ap.add_argument("--out", default="results/cifar100/canonical_partition.csv")
    ap.add_argument("--schedule-out", default="results/cifar100/canonical_round_schedule.csv")
    args = ap.parse_args()

    # Seed exactly as train.py does before _precompute_client_budgets.
    import random, torch
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    budgets = precompute_budgets(args.n_clients, args.min_mac, args.max_mac, args.zipf_alpha)

    out = partition_data(
        dataset="cifar100", datadir=args.data_dir, partition="hetero",
        n_nets=args.n_clients, alpha=args.partition_alpha, load_from_pkl=False,
        validation_split=args.val_split,
        client_budgets=budgets, corr_gamma=1.0, max_training_mac=args.cap_mac,
    )
    net_dataidx_map = out[6]

    cache = load_cache(REPO / args.cache)
    macs = np.array([c["macs"] for c in cache])
    cap = args.cap_mac
    envelope_ids = {}          # (tuple max_d, tuple max_w) -> integer id
    rows = []
    for i in range(args.n_clients):
        n_i = len(net_dataidx_map[i])
        b = budgets[i]
        n_aff = int((macs <= min(b, cap)).sum()) or 1
        afford = cache[:n_aff]                       # cache sorted ascending by MAC
        local_max_idx = n_aff - 1                    # affordable-cache index of local-max
        max_d, max_w = routed_envelope(afford)
        key = (tuple(max_d), tuple(max_w))
        env_id = envelope_ids.setdefault(key, len(envelope_ids))
        rows.append({
            "client": i,
            "n_i": n_i,
            "B_i_batches": -(-n_i // args.batch_size),   # ceil, drop_last=False
            "budget_M": round(b / 1e6, 2),
            "capped_budget_M": round(min(b, cap) / 1e6, 2),
            "routed_local_max_M": round(cache[local_max_idx]["macs"] / 1e6, 2),
            "n_affordable": n_aff,                        # |affordable cache set|
            "local_max_cache_idx": local_max_idx,        # index into ascending cache
            "envelope_id": env_id,                       # unique routed-envelope group
            "envelope_max_d": str(max_d),                # per-stage routed depth bound
            "envelope_max_w_indices": str(max_w),        # per-position routed width bound
        })

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        w.writeheader(); w.writerows(rows)

    # ---- per-round schedule (deterministic: seed=round_idx client sampling) ----
    # _client_sampling: np.random.seed(round_idx); choice(N, m, replace=False).
    # eta_t = base_lr * 0.5 * (1 + cos(pi t / T)).
    T = args.rounds
    by_client = {r["client"]: r for r in rows}
    sched_rows = []
    for t in range(T):
        eta_t = args.lr * 0.5 * (1.0 + np.cos(np.pi * t / T))
        np.random.seed(t)
        sel = np.random.choice(range(args.n_clients), args.clients_per_round, replace=False)
        for cid in sel:
            r = by_client[int(cid)]
            sched_rows.append({
                "round": t, "eta_t": eta_t, "client": int(cid),
                "n_i": r["n_i"], "B_i": r["B_i_batches"],
                "envelope_id": r["envelope_id"],
                "routed_local_max_M": r["routed_local_max_M"],
            })
    sched_out = REPO / args.schedule_out
    with open(sched_out, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(sched_rows[0].keys()), lineterminator="\n"
        )
        w.writeheader(); w.writerows(sched_rows)

    n = np.array([r["n_i"] for r in rows])
    B = np.array([r["B_i_batches"] for r in rows])
    print(f"clients={args.n_clients} seed={args.seed} val_split={args.val_split} batch={args.batch_size}")
    print(f"n_i:  min={n.min()} max={n.max()} mean={n.mean():.1f} median={int(np.median(n))} total={n.sum()}")
    print(f"B_i (ceil, drop_last=False): min={B.min()} max={B.max()} mean={B.mean():.1f} "
          f"median={int(np.median(B))} total_steps/round={B.sum()}")
    n_env = len(set(r["envelope_id"] for r in rows))
    print(f"routed envelopes (unique profiles): {n_env}")
    # applicable-step scale per envelope: sum of B_i over its clients, weighted by
    # participation (each client sampled ~ clients_per_round/n_clients of rounds).
    print(f"schedule rows: {len(sched_rows)} (= rounds x clients_per_round)")
    print("n_i list:", n.tolist())
    print(f"\nPartition CSV -> {outp}")
    print(f"Round schedule CSV -> {REPO / args.schedule_out}")


if __name__ == "__main__":
    main()
