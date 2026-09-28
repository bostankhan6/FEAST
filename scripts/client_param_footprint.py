#!/usr/bin/env python3
"""
client_param_footprint.py - Cross-method client-side parameter footprint at
matched compute budgets.

Answers the "how much does each method actually send over the wire" question:
at a shared MAC budget, how many parameters does each method's client-received
model contain? This is the basis for any bandwidth/communication-cost claim
comparing FEAST against HeteroFL/ScaleFL/FIARSE.

For FEAST, the client-received model is the routed sub-supernet — the
coordinate-wise envelope (max depth per stage, max width index per position)
over every cached subnet affordable at the client's budget (Supplementary Eq.
sub_supernet_bounds). This mirrors GenericServerOFA.compute_sub_supernet_bounds
in src/feast/Server/generic_server_model.py (kept as a pure/standalone
reimplementation here since this script builds architecture-only models,
without server weights). The five canonical CIFAR-100 budget anchors this
script reports at (25M/50M/229M/408M/596M) match Supplementary Table
training_envelope_params / deployed_params.

Usage:
    ./venv/bin/python3 scripts/client_param_footprint.py \
        --out results/cifar100/client_param_footprint.csv
"""
import argparse
import ast
import csv
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from feast.elastic_nn.generic_ofa_network import GenericOFAResNet
from baselines.heterofl.federation import budget_to_tier, TIER_RATES
from baselines.heterofl.model import HeteroFLResNet
from baselines.scalefl.federation import budget_to_level, LEVEL_MACS
from baselines.scalefl.model import build_client_model as scalefl_client
from baselines.fiarse.model import build_global_model as fiarse_build_global, FULL_MODEL_MACS

ANCHORS_M = [25, 50, 229, 408, 596]


def load_cache(path, exp_choices):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            e = ast.literal_eval(r["e"])
            rows.append({
                "macs": float(r["macs"]),
                "d": ast.literal_eval(r["d"]),
                "e_idx": [min(range(len(exp_choices)),
                              key=lambda i: abs(exp_choices[i] - ev)) for ev in e],
                "w_indices": ast.literal_eval(r["w_indices"]),
            })
    rows.sort(key=lambda x: x["macs"])
    return rows


def routed_envelope(cache, budget, cap=600e6):
    """Coordinate-wise max over every cached subnet <= min(budget, cap) --
    mirrors Server/generic_server_model.py:compute_sub_supernet_bounds."""
    afford = [c for c in cache if c["macs"] <= min(budget, cap)] or cache[:1]
    nd, nw = len(afford[0]["d"]), len(afford[0]["w_indices"])
    max_d = [max(c["d"][s] for c in afford) for s in range(nd)]
    max_w = [max(c["w_indices"][p] for c in afford) for p in range(nw)]
    return max_d, max_w, len(afford)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/supernets/4-stage-supernet-cifar100-v2.json")
    ap.add_argument("--cache", default="subnet_caches/extended_range_25M_1500M.csv")
    ap.add_argument("--n-classes", type=int, default=100)
    ap.add_argument("--anchors-m", type=float, nargs="+", default=ANCHORS_M,
                     help="Client MAC budgets (in millions) to compare at.")
    ap.add_argument("--out", default="results/cifar100/client_param_footprint.csv")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(REPO, args.config)))
    cfg["n_classes"] = args.n_classes
    cache = load_cache(os.path.join(REPO, args.cache), cfg["expansion_ratio_choices"])
    sub_params = {k: v for k, v in cfg.items() if k not in ("alpha_weights", "beta_depth_penalty")}

    full_sn = GenericOFAResNet(**sub_params)
    full_sn_params = n_params(full_sn)

    fiarse_gm = fiarse_build_global(num_classes=args.n_classes)
    fiarse_params = n_params(fiarse_gm)  # constant at every budget: unstructured masking, dense tensor always transmitted

    print(f"Full FEAST supernet: {full_sn_params/1e6:.2f}M params")
    print(f"FIARSE global model (constant, unstructured masking): {fiarse_params/1e6:.2f}M params\n")

    rows = []
    header = f"{'budget_M':>8}  {'feast_subsn_M':>14}  {'heterofl_M':>11}  {'scalefl_M':>10}  {'fiarse_M':>9}  largest"
    print(header)
    for b_m in args.anchors_m:
        b = b_m * 1e6

        max_d, max_w, n_aff = routed_envelope(cache, b)
        feast_sub = GenericOFAResNet(per_position_max_w_indices=max_w, per_position_max_d=max_d, **sub_params)
        feast_p = n_params(feast_sub)

        tier = budget_to_tier(b)
        hfl_p = n_params(HeteroFLResNet(model_rate=TIER_RATES[tier], num_classes=args.n_classes))

        level = budget_to_level(b)
        sfl_p = n_params(scalefl_client(level=level, num_classes=args.n_classes))

        vals = {"FEAST": feast_p, "HeteroFL": hfl_p, "ScaleFL": sfl_p, "FIARSE": fiarse_params}
        largest = max(vals, key=vals.get)

        print(f"{b_m:>8}  {feast_p/1e6:>14.2f}  {hfl_p/1e6:>11.2f}  {sfl_p/1e6:>10.2f}  "
              f"{fiarse_params/1e6:>9.2f}  {largest}")

        rows.append({
            "budget_m": b_m,
            "feast_subsn_params_m": round(feast_p / 1e6, 3),
            "feast_n_affordable": n_aff,
            "heterofl_tier": tier, "heterofl_rate": TIER_RATES[tier],
            "heterofl_params_m": round(hfl_p / 1e6, 3),
            "scalefl_level": level,
            "scalefl_params_m": round(sfl_p / 1e6, 3),
            "fiarse_params_m": round(fiarse_params / 1e6, 3),
            "largest_method": largest,
        })

    outp = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nFull FEAST supernet: {full_sn_params/1e6:.2f}M")
    print(f"CSV -> {outp}")


if __name__ == "__main__":
    main()
