#!/usr/bin/env python
"""
reproduce_sub_supernet_communication_reduction.py

Validates the routed sub-supernet mechanism (component 3 in README.md,
"Sub-supernet communication") under the canonical Zipf(alpha) client-budget
distribution, and reports the population-weighted communication reduction
this produces vs. broadcasting the full supernet to every client — the
number this repo's paper cites for that claim.

Answers a different question than scripts/client_param_footprint.py (which
compares FEAST's client footprint against HeteroFL/ScaleFL/FIARSE at shared
budgets). This script compares FEAST's routed sub-supernet against FEAST's
OWN full supernet, population-weighted over Zipf, using the real cache and
config the paper's canonical CIFAR-100 run uses (see
src/feast/Server/generic_server_model.py's compute_sub_supernet_bounds /
create_sub_supernet for the routing implementation).

Checks:
  A) Every subnet in the sub-cache can actually be activated in its
     corresponding routed sub-supernet (no ValueError / shape mismatch).
  B) Per-budget-tier parameter count and the resulting communication
     reduction vs the full supernet, weighted by how many of the N sampled
     clients land in each tier.
  C) Headroom: the routed sub-supernet's bounds are exactly as tight as the
     affordable subnets require (zero slack) -- proves the client isn't
     being sent parameters no affordable subnet actually needs.
  D) Forward-pass spot check across a sample of budget tiers.

Usage:
    python scripts/reproduce_sub_supernet_communication_reduction.py
    python scripts/reproduce_sub_supernet_communication_reduction.py --synthetic
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from feast.Server.generic_server_model import GenericServerOFA

NUM_STAGES = 4

REAL_CONFIG_PATH = os.path.join(REPO, "configs", "supernets", "4-stage-supernet-cifar100-v2.json")
REAL_CACHE_PATH = os.path.join(
    REPO, "subnet_caches", "extended_range_25M_1500M.csv"
)
REAL_MAX_MAC = 1500e6


def make_e_values(ratio):
    return [ratio] * NUM_STAGES


def precompute_client_budgets(n_clients, min_mac, max_mac, alpha, n_force_max=0):
    """Exact copy of train.py:_precompute_client_budgets's zipf branch. Used
    instead of GenericServerOFA.assign_client_resources()'s own zipf branch,
    which draws an extra (unused) np.random.zipf() sample first and so
    consumes a different RNG stream -- passing pre-computed budgets in here
    (like train.py does for real --weight_dataset_by_budget runs) is what
    makes this script's output match the paper's canonical run."""
    ranks = np.arange(1, n_clients + 1)
    probs = 1.0 / np.power(ranks, alpha)
    probs /= probs.sum()
    level_indices = np.random.choice(ranks, size=n_clients, p=probs)
    steps = (max_mac - min_mac) / max(1, (n_clients - 1))
    budgets = min_mac + (level_indices - 1) * steps
    budgets += np.random.uniform(-steps / 2, steps / 2, size=n_clients)
    budgets = np.clip(budgets, min_mac, max_mac)
    if n_force_max > 0:
        for i in range(min(n_force_max, n_clients)):
            budgets[i] = max_mac
    np.random.shuffle(budgets)
    return {i: float(budgets[i]) for i in range(n_clients)}


def build_synthetic_cache_df():
    """10 synthetic configs spanning 10M-6B MACs so Zipf creates many
    distinct tiers. For fast, CSV-independent correctness testing only --
    not representative of the paper's actual communication-reduction number."""
    records = [
        {'macs': 1e7,  'params': 0, 'd': str([0,0,0,0]), 'e': str(make_e_values(0.10)), 'w_indices': str([3,0,0,0,0])},
        {'macs': 4e7,  'params': 0, 'd': str([0,0,0,1]), 'e': str(make_e_values(0.14)), 'w_indices': str([5,1,1,1,1])},
        {'macs': 8e7,  'params': 0, 'd': str([0,0,0,2]), 'e': str(make_e_values(0.14)), 'w_indices': str([5,1,1,1,2])},
        {'macs': 1.5e8,'params': 0, 'd': str([0,0,1,2]), 'e': str(make_e_values(0.14)), 'w_indices': str([5,1,1,2,2])},
        {'macs': 2.5e8,'params': 0, 'd': str([0,1,1,2]), 'e': str(make_e_values(0.18)), 'w_indices': str([6,1,2,3,3])},
        {'macs': 4e8,  'params': 0, 'd': str([1,1,2,2]), 'e': str(make_e_values(0.18)), 'w_indices': str([6,2,2,3,4])},
        {'macs': 6e8,  'params': 0, 'd': str([1,1,2,3]), 'e': str(make_e_values(0.18)), 'w_indices': str([7,2,3,4,5])},
        {'macs': 1.5e9,'params': 0, 'd': str([2,2,3,4]), 'e': str(make_e_values(0.22)), 'w_indices': str([8,4,5,6,7])},
        {'macs': 3e9,  'params': 0, 'd': str([3,3,4,5]), 'e': str(make_e_values(0.22)), 'w_indices': str([8,5,6,7,8])},
        {'macs': 6e9,  'params': 0, 'd': str([5,5,5,5]), 'e': str(make_e_values(0.25)), 'w_indices': str([9,7,8,8,9])},
    ]
    return pd.DataFrame(records)


def build_server(use_real_cache=True, num_clients=100, zipf_alpha=1.2, force_max=0, init_seed=0):
    config_path = REAL_CONFIG_PATH
    with open(config_path) as f:
        cfg = json.load(f)
    cfg['n_classes'] = 100  # CIFAR-100

    cfg['resource_heterogeneity'] = True
    cfg['resource_distribution_type'] = 'zipf'
    cfg['resource_zipf_alpha'] = zipf_alpha
    cfg['resource_force_max_clients'] = force_max

    pre_computed_budgets = None
    if use_real_cache:
        cfg['subnet_cache_path'] = REAL_CACHE_PATH
        cfg['resource_max_mac'] = REAL_MAX_MAC
        print(f"  [Mode] REAL cache: {os.path.basename(REAL_CACHE_PATH)}")
        print(f"         Config:     {os.path.basename(REAL_CONFIG_PATH)}")
        print(f"         Max budget: {REAL_MAX_MAC/1e6:.0f}M MACs")

        cache_df = pd.read_csv(REAL_CACHE_PATH)
        min_mac = float(cache_df['macs'].min())

        np.random.seed(init_seed)
        pre_computed_budgets = precompute_client_budgets(
            num_clients, min_mac, REAL_MAX_MAC, zipf_alpha, n_force_max=force_max
        )
    else:
        df = build_synthetic_cache_df()
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        df.to_csv(tmp.name, index=False)
        tmp.close()
        cfg['subnet_cache_path'] = tmp.name
        cfg.pop('resource_max_mac', None)
        print(f"  [Mode] SYNTHETIC cache (10M-6B MACs)")
        np.random.seed(init_seed)

    return GenericServerOFA(
        arch_params=cfg,
        sampling_method='TS_optimal_path',
        num_cli_total=num_clients,
        client_mac_budgets=pre_computed_budgets,
    )


# ---------------------------------------------------------------------------
# A) Correctness: every sub_cache config must be settable in the sub-supernet
# ---------------------------------------------------------------------------
def validate_all_configs(server, num_clients):
    print("\n" + "=" * 70)
    print("A) CORRECTNESS: activating every sub_cache config in each sub-supernet")
    print("=" * 70)

    expansion_choices = server.arch_params['expansion_ratio_choices']

    failures = []
    total_checks = 0
    unique_budgets = {}  # budget_key -> (sub_sn, sub_info)

    for cli_id in range(num_clients):
        budget = server.client_mac_budgets[cli_id]
        budget_key = round(budget / 1e6, 2)

        if budget_key not in unique_budgets:
            sub_sn, sub_info = server.create_sub_supernet(budget)
            unique_budgets[budget_key] = (sub_sn, sub_info)
        else:
            sub_sn, sub_info = unique_budgets[budget_key]

        for cfg_idx, arch in enumerate(sub_info['sub_cache']):
            total_checks += 1
            try:
                e_indices = [expansion_choices.index(e) for e in arch['e']]
            except ValueError as exc:
                failures.append((cli_id, cfg_idx, f"e_index lookup: {exc}"))
                continue
            try:
                sub_sn.set_active_subnet(
                    d=arch['d'],
                    e_indices=e_indices,
                    w_indices=arch['w_indices'],
                )
            except Exception as exc:
                failures.append((cli_id, cfg_idx,
                                  f"set_active_subnet failed: {exc} | "
                                  f"w_indices={arch['w_indices']}"))

    print(f"  Total (client x config) checks : {total_checks}")
    print(f"  Unique budget tiers tested     : {len(unique_budgets)}")
    if not failures:
        print(f"  Result                         : ALL PASSED")
    else:
        print(f"  FAILURES ({len(failures)}):")
        for cli_id, cfg_idx, msg in failures[:20]:
            print(f"    client {cli_id}, config {cfg_idx}: {msg}")
    return failures, unique_budgets


# ---------------------------------------------------------------------------
# B) Efficiency analysis
# ---------------------------------------------------------------------------
def analyze_efficiency(server, unique_budgets, num_clients):
    print("\n" + "=" * 70)
    print("B) EFFICIENCY: parameter / communication savings per budget tier")
    print("=" * 70)

    global_params = sum(p.numel() for p in server.model.parameters())
    print(f"\n  Full supernet: {global_params/1e6:.2f}M params\n")

    header = (f"  {'Budget(M)':>10} | {'#Clients':>8} | {'#Configs':>8} | "
              f"{'SubSN(M)':>9} | {'Reduction':>10} | "
              f"{'StemMax':>8} | {'S0':>5} | {'S1':>5} | {'S2':>5} | {'S3':>5} |"
              f" {'MaxD':>15}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    budget_client_count = {}
    for cid in range(num_clients):
        key = round(server.client_mac_budgets[cid] / 1e6, 2)
        budget_client_count[key] = budget_client_count.get(key, 0) + 1

    rows = []
    w_choices = server.arch_params['width_multiplier_choices']
    for budget_key, (sub_sn, sub_info) in sorted(unique_budgets.items()):
        sub_p = sum(p.numel() for p in sub_sn.parameters())
        ratio = global_params / sub_p
        bounds = sub_info['bounds']
        mw = bounds['max_w_indices']
        max_d = bounds['max_d']
        n_cli = budget_client_count.get(budget_key, 0)
        n_cfg = len(sub_info['sub_cache'])
        rows.append((budget_key, n_cli, n_cfg, sub_p, ratio, mw, max_d))

        stage_maxes = [f"{w_choices[mw[i+1]]:.1f}" for i in range(4)]
        print(f"  {budget_key:>10.2f} | {n_cli:>8d} | {n_cfg:>8d} | "
              f"{sub_p/1e6:>9.2f} | {ratio:>9.2f}x | "
              f"{w_choices[mw[0]]:>8.1f} | " + " | ".join(f"{s:>5}" for s in stage_maxes)
              + f" | {str(max_d):>15}")

    print()
    # Per-client reduction ratios (each tier's ratio repeated once per client in that tier) --
    # this is the distribution the paper's "Mean/Median/Largest client reduction" rows summarize.
    per_client_ratios = []
    for r in rows:
        per_client_ratios.extend([r[4]] * r[1])
    per_client_ratios.sort()

    mean_client_ratio = sum(per_client_ratios) / num_clients
    n = len(per_client_ratios)
    median_client_ratio = (
        per_client_ratios[n // 2] if n % 2 == 1
        else (per_client_ratios[n // 2 - 1] + per_client_ratios[n // 2]) / 2
    )
    max_client_ratio = per_client_ratios[-1]

    weighted_sub_p = sum(r[1] * r[3] for r in rows) / num_clients
    # Aggregate reduction is the paper's R^comm_t = m*P / sum_i P_i (ratio of SUMS, i.e.
    # full-supernet payload sent to every client vs the sum of routed payloads actually sent)
    # -- distinct from mean_client_ratio (mean of per-client ratios P/P_i). These differ
    # whenever budgets are heterogeneous (arithmetic mean of ratios != ratio of means).
    aggregate_ratio = global_params / weighted_sub_p

    print(f"  Total traffic reduction  (aggregate, mP / sum P_i) : {aggregate_ratio:.2f}x")
    print(f"  Mean client reduction    (mean of P / P_i)         : {mean_client_ratio:.2f}x")
    print(f"  Median client reduction                            : {median_client_ratio:.2f}x")
    print(f"  Largest client reduction                           : {max_client_ratio:.2f}x")
    print(f"  Mean routed sub-supernet size                      : {weighted_sub_p/1e6:.2f}M params")
    print(f"  (vs full supernet                                  : {global_params/1e6:.2f}M params)")

    print("\n  Budget distribution (bins):")
    budgets_all = [server.client_mac_budgets[i] / 1e6 for i in range(num_clients)]
    actual_macs = server.subnet_cache_macs
    bin_edges = [m / 1e6 for m in actual_macs]
    counts = [0] * len(bin_edges)
    for b in budgets_all:
        for i, edge in enumerate(bin_edges):
            if b <= edge:
                counts[i] += 1
                break
    for edge, cnt in zip(bin_edges, counts):
        bar = "#" * cnt
        print(f"    <={edge:7.1f}M: {cnt:3d} |{bar}")

    return rows, {
        'aggregate_ratio': aggregate_ratio,
        'mean_client_ratio': mean_client_ratio,
        'median_client_ratio': median_client_ratio,
        'max_client_ratio': max_client_ratio,
    }


# ---------------------------------------------------------------------------
# C) Headroom: bounds should be tight (headroom == 0 for all positions)
# ---------------------------------------------------------------------------
def analyze_headroom(server, unique_budgets):
    print("\n" + "=" * 70)
    print("C) HEADROOM: gap between sub_cache max_w and sub-supernet bound")
    print("   (All should be 0 -- proves bounds are exactly tight)")
    print("=" * 70)

    any_nonzero = False
    w_choices = server.arch_params['width_multiplier_choices']
    positions = ['stem', 's0  ', 's1  ', 's2  ', 's3  ']

    for budget_key, (sub_sn, sub_info) in sorted(unique_budgets.items()):
        bounds = sub_info['bounds']
        mw = bounds['max_w_indices']
        sub_cache = sub_info['sub_cache']
        if not sub_cache:
            continue

        actual_max = [max(cfg['w_indices'][p] for cfg in sub_cache) for p in range(5)]
        headroom = [mw[i] - actual_max[i] for i in range(5)]

        if any(h != 0 for h in headroom):
            any_nonzero = True
            print(f"\n  Budget {budget_key:.2f}M -- headroom {headroom} (non-zero = looser than needed):")
            for pos_name, h, am, bm in zip(positions, headroom, actual_max, mw):
                print(f"    {pos_name}: cache_max={am} ({w_choices[am]:.1f}x), "
                      f"sn_bound={bm} ({w_choices[bm]:.1f}x), headroom={h}")

    if not any_nonzero:
        print("  All headroom values are 0 -- sub-supernet bounds are exactly tight.")
    return not any_nonzero


# ---------------------------------------------------------------------------
# D) Forward-pass spot check across budget tiers
# ---------------------------------------------------------------------------
def spot_check_forward(server, unique_budgets):
    print("\n" + "=" * 70)
    print("D) FORWARD PASS: spot check min/max config for representative tiers")
    print("=" * 70)

    expansion_choices = server.arch_params['expansion_ratio_choices']
    x = torch.randn(2, 3, 32, 32)

    sorted_tiers = sorted(unique_budgets.keys())
    step = max(1, len(sorted_tiers) // 5)
    sample_tiers = sorted_tiers[::step][:5]

    all_ok = True
    for budget_key in sample_tiers:
        sub_sn, sub_info = unique_budgets[budget_key]
        if next(sub_sn.parameters()).is_cuda:
            x = x.cuda()

        sub_cache = sub_info['sub_cache']
        if not sub_cache:
            continue

        for label, arch in [('min', sub_cache[0]), ('max', sub_cache[-1])]:
            e_idx = [expansion_choices.index(e) for e in arch['e']]
            sub_sn.set_active_subnet(d=arch['d'], e_indices=e_idx,
                                      w_indices=arch['w_indices'])
            y = sub_sn(x)
            ok = (y.shape == (2, server.arch_params['n_classes']))
            if not ok:
                all_ok = False
            print(f"  Budget {budget_key:8.1f}M [{label}]: "
                  f"d={arch['d']}  out={tuple(y.shape)} {'OK' if ok else 'FAIL'}")

    if all_ok:
        print("  All forward passes succeeded.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true",
                         help="Use a fast synthetic cache instead of the real canonical cache "
                              "(for quick correctness checks; does not reproduce the paper's number).")
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--force-max-clients", type=int, default=0,
                         help="Extra clients forced to the max budget, on top of the Zipf draw. "
                              "0 matches the paper's canonical population (Zipf(1.2) only, no forcing).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_real_cache = not args.synthetic
    mode_label = "SYNTHETIC cache" if args.synthetic else "REAL cache (25M-1.5G budgets, canonical 56-subnet cache)"
    print("=" * 70)
    print(f"Sub-Supernet Zipf Validation ({args.num_clients} clients, alpha={args.zipf_alpha})")
    print(f"Mode: {mode_label}")
    print("=" * 70)

    server = build_server(use_real_cache=use_real_cache, num_clients=args.num_clients,
                           zipf_alpha=args.zipf_alpha, force_max=args.force_max_clients,
                           init_seed=args.seed)

    print(f"\n  Cache MACs (actual, recalculated at runtime):")
    for i, m in enumerate(server.subnet_cache_macs):
        print(f"    [{i}] {m/1e6:.2f}M  d={server.subnet_cache[i]['d']}")

    failures, unique_budgets = validate_all_configs(server, args.num_clients)
    rows, reduction_stats = analyze_efficiency(server, unique_budgets, args.num_clients)
    headroom_ok = analyze_headroom(server, unique_budgets)
    fwd_ok = spot_check_forward(server, unique_budgets)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not failures and fwd_ok:
        print("  ALL correctness checks passed")
        print(f"  Total traffic reduction (aggregate) : {reduction_stats['aggregate_ratio']:.2f}x")
        print(f"  Mean client reduction                : {reduction_stats['mean_client_ratio']:.2f}x")
        print(f"  Median client reduction               : {reduction_stats['median_client_ratio']:.2f}x")
        print(f"  Largest client reduction              : {reduction_stats['max_client_ratio']:.2f}x")
        print("  All forward passes passed")
    else:
        if failures:
            print(f"  {len(failures)} set_active_subnet failures -- see above")
        if not fwd_ok:
            print("  Forward pass failures -- see above")
    print("=" * 70)

    if failures or not fwd_ok or not headroom_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
