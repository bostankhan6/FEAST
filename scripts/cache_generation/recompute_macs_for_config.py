"""
Re-compute MACs and params for an existing subnet cache under a different
supernet config (e.g., different input resolution or stem stride).

Use case: produce a TinyImageNet (64x64, stem_stride=2) cache that has the
SAME architectures as the CIFAR/CINIC cache but with MAC numbers reflecting
the new input shape. This keeps cross-dataset comparisons clean.

Usage:
    ./venv/bin/python scripts/cache_generation/recompute_macs_for_config.py \
        --input_csv  subnet_caches/extended_range_25M_1500M.csv \
        --config     configs/supernets/4-stage-supernet-tinyimagenet.json \
        --output_csv subnet_caches/extended_range_tinyimagenet_25M_1500M.csv
"""

import argparse
import ast
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from feast.utils.subnet_cost import subnet_macs


def parse_list(val):
    """Parse a list-as-string from CSV (e.g., '[3, 7, 6, 7]') into a Python list."""
    if isinstance(val, list):
        return val
    return ast.literal_eval(val)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_csv", required=True, help="Source cache CSV (architectures kept verbatim)")
    parser.add_argument("--config", required=True, help="Supernet config JSON to evaluate MACs against")
    parser.add_argument("--output_csv", required=True, help="Destination CSV with re-computed MACs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        arch_config = json.load(f)
    width_choices = arch_config["width_multiplier_choices"]

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} subnets from {args.input_csv}")
    print(f"Using config: {args.config}")
    print(
        f"  initial_input_hw={arch_config['initial_input_hw']}, "
        f"stem_stride={arch_config.get('stem_stride', 1)}, "
        f"n_classes={arch_config['n_classes']}"
    )

    new_macs, new_params = [], []
    for _, row in df.iterrows():
        d = parse_list(row["d"])
        e = parse_list(row["e"])
        w = parse_list(row["w_indices"])
        macs, params = subnet_macs(d, e, w, width_mult_options=width_choices, arch_config_params=arch_config)
        new_macs.append(macs)
        new_params.append(params)

    out = df.copy()
    out["macs_original"] = df["macs"]
    out["params_original"] = df["params"]
    out["macs"] = new_macs
    out["params"] = new_params

    out = out[
        ["macs", "params", "macs_original", "params_original"]
        + [c for c in df.columns if c not in ("macs", "params")]
    ]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(out)} rows -> {args.output_csv}")

    print("\nFirst 5 rows (MAC delta vs original):")
    for i in range(min(5, len(out))):
        old = out["macs_original"].iloc[i]
        new = out["macs"].iloc[i]
        delta = (new - old) / old * 100
        print(f"  idx={i}  old={old/1e6:7.2f}M  new={new/1e6:7.2f}M  delta={delta:+.2f}%")

    print(
        f"\nMAC range: [{out['macs'].min()/1e6:.2f}M, {out['macs'].max()/1e6:.2f}M]"
    )


if __name__ == "__main__":
    main()
