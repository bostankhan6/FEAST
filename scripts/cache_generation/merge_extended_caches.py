#!/usr/bin/env python3
"""
Merge extended range cache files into a single consolidated cache.

Usage:
    python merge_extended_caches.py \
        --low_end extended_range_low_10M_50M.csv \
        --original 8_blocks_supernet_cache_alpha_weights_all_1s_rho_0-31.csv \
        --output extended_range_10M_1500M.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def merge_caches(low_csv, original_csv, output_csv):
    """Merge low-end and original cache files, sort by MACs."""

    print(f"Reading low-end cache: {low_csv}")
    low_df = pd.read_csv(low_csv)
    # Convert macs to millions for display
    low_macs_m = low_df['macs'] / 1_000_000
    print(f"  Entries: {len(low_df)}, MACs range: [{low_macs_m.min():.1f}M, {low_macs_m.max():.1f}M]")

    print(f"Reading original cache: {original_csv}")
    orig_df = pd.read_csv(original_csv)
    # Convert macs to millions for display
    orig_macs_m = orig_df['macs'] / 1_000_000
    print(f"  Entries: {len(orig_df)}, MACs range: [{orig_macs_m.min():.1f}M, {orig_macs_m.max():.1f}M]")

    # Merge
    merged_df = pd.concat([low_df, orig_df], ignore_index=True)

    # Sort by MACs
    merged_df = merged_df.sort_values('macs').reset_index(drop=True)
    # Add macs_m column for display (millions)
    merged_df['macs_m'] = merged_df['macs'] / 1_000_000

    print(f"\nMerged cache: {len(merged_df)} entries")
    print(f"  MACs range: [{merged_df['macs_m'].min():.1f}M, {merged_df['macs_m'].max():.1f}M]")
    print(f"  Columns: {list(merged_df.columns)}")

    # Check for duplicates in MACs (tolerance: 0.1M)
    macs_vals = merged_df['macs_m'].values
    diffs = np.diff(macs_vals)
    close_pairs = np.sum(diffs < 0.1)
    if close_pairs > 0:
        print(f"\n  WARNING: {close_pairs} pairs of subnets with MACs within 0.1M of each other")
        print("  These may be near-duplicates from different cache generations")

    # Save (drop the macs_m column before saving, keep only original columns)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save without macs_m (keep original structure)
    merged_df.drop('macs_m', axis=1).to_csv(output_csv, index=False)
    print(f"\nMerged cache saved to: {output_csv}")

    # Summary by MAC tier
    print("\nSummary by MAC tier:")
    tiers = [(10, 50), (50, 100), (100, 200), (200, 400), (400, 600), (600, 1500)]
    for lo, hi in tiers:
        count = ((merged_df['macs_m'] >= lo) & (merged_df['macs_m'] < hi)).sum()
        if count > 0:
            print(f"  [{lo:4d}M, {hi:4d}M): {count:2d} subnets")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge extended range cache files")
    parser.add_argument('--low_end', required=True, help='Low-end cache CSV (10M-50M)')
    parser.add_argument('--original', required=True, help='Original cache CSV (50M-600M)')
    parser.add_argument('--output', required=True, help='Output merged cache CSV')

    args = parser.parse_args()
    merge_caches(args.low_end, args.original, args.output)
