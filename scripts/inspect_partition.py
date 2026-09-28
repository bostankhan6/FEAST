#!/usr/bin/env python
"""Print per-client class counts for a Dirichlet partition of CIFAR-100.

Usage:
    ./venv/bin/python scripts/inspect_partition.py --alpha 0.3
    ./venv/bin/python scripts/inspect_partition.py --alpha 0.05 --n_clients 100 --seed 0
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))

from feast.data.cifar100.data_loader import partition_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data/cifar100")
    p.add_argument("--validation_split", type=float, default=0.1)
    p.add_argument("--show_clients", type=int, default=10,
                   help="How many clients to print in detail (sorted by sample count).")
    args = p.parse_args()

    np.random.seed(args.seed)

    out = partition_data(
        dataset="cifar100",
        datadir=args.data_dir,
        partition="hetero",
        n_nets=args.n_clients,
        alpha=args.alpha,
        load_from_pkl=False,
        validation_split=args.validation_split,
    )
    # Returns: X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts
    net_dataidx_map = out[6]
    cls_counts = out[7]

    sizes = np.array([len(net_dataidx_map[i]) for i in range(args.n_clients)])
    n_classes_per_client = np.array([len(cls_counts[i]) for i in range(args.n_clients)])

    print("=" * 60)
    print(f"Dirichlet alpha={args.alpha}, n_clients={args.n_clients}, seed={args.seed}")
    print("=" * 60)
    print(f"Samples per client    | min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.1f} median={int(np.median(sizes))}")
    print(f"Classes per client    | min={n_classes_per_client.min()} "
          f"max={n_classes_per_client.max()} mean={n_classes_per_client.mean():.1f} "
          f"median={int(np.median(n_classes_per_client))}")
    print(f"Total samples         | {sizes.sum()}")
    print()

    # Histogram of classes/client
    buckets = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
    print("Histogram (clients by #classes):")
    for lo, hi in buckets:
        n = int(((n_classes_per_client >= lo) & (n_classes_per_client <= hi)).sum())
        print(f"  {lo:>3}–{hi:<3} classes: {n} clients")
    print()

    # Show a few clients
    order = np.argsort(-sizes)
    show = order[:args.show_clients]
    print(f"Top-{args.show_clients} clients by sample count:")
    for cid in show:
        cls = cls_counts[int(cid)]
        top5 = sorted(cls.items(), key=lambda kv: -kv[1])[:5]
        top5_str = ", ".join(f"{k}:{v}" for k, v in top5)
        print(f"  client {int(cid):3d} | n={sizes[cid]:5d} | "
              f"#classes={len(cls):3d} | top5: {top5_str}")


if __name__ == "__main__":
    main()
