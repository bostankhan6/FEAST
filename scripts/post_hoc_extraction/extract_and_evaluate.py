#!/usr/bin/env python
"""
extract_and_evaluate.py - Extract, BN-recalibrate, and evaluate a post-hoc
subnet architecture from a trained FEAST checkpoint.

Takes an architecture found by run_single_subnet_search.sh (a JSON file with
d/e/w_indices, as produced by src/feast/nas/search_single_subnet.py) and:
  1. Loads the trained supernet checkpoint and activates that architecture
     via set_active_subnet (this IS the extraction -- FEAST's elastic layers
     slice weights for whichever architecture is active, no separate copy
     step is needed).
  2. BN-recalibrates on the same held-out calibration split evaluate.py uses
     (no fine-tuning or further training).
  3. Evaluates on the official test set.
  4. Optionally reports the gap against a reference cached-variant curve
     (linear interpolation at the same realized MAC value), matching the
     paper's post-hoc extraction gap metric.

Usage:
    # Single architecture:
    python scripts/post_hoc_extraction/extract_and_evaluate.py \
        --checkpoint checkpoints/feast-cifar100-mixaug/best_checkpoint_supernet.pt \
        --dataset cifar100 \
        --arch_json subnet_caches/posthoc/target_30000000.0.json \
        --augmentation mixaug

    # Every found architecture in a directory (batch, e.g. a MAC sweep):
    python scripts/post_hoc_extraction/extract_and_evaluate.py \
        --checkpoint checkpoints/feast-cifar100-mixaug/best_checkpoint_supernet.pt \
        --dataset cifar100 \
        --arch_json_dir subnet_caches/posthoc/ \
        --augmentation mixaug \
        --reference_curve results/cifar100/feast_eval_full56.csv \
        --output results/cifar100/posthoc_eval.csv
"""
import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from evaluate import (  # noqa: E402
    load_test_data,
    load_bn_calibration_data,
    evaluate_subnet,
    _reseed_bn_cal,
)
from feast.Server.generic_server_model import GenericServerOFA  # noqa: E402
from feast.utils.subnet_cost import subnet_macs  # noqa: E402
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_arch_json(path):
    with open(path) as f:
        data = json.load(f)
    return {"d": data["d"], "e": data["e"], "w_indices": data["w_indices"]}, data


def e_values_to_indices(e_values, expansion_choices):
    """Map each expansion ratio to its closest index in the search space
    (robust to JSON float round-tripping, unlike an exact equality lookup)."""
    return [min(range(len(expansion_choices)), key=lambda i: abs(expansion_choices[i] - ev))
            for ev in e_values]


def interpolate_reference_accuracy(reference_curve_path, macs_m):
    df = pd.read_csv(reference_curve_path).sort_values("macs_m").reset_index(drop=True)
    return float(np.interp(macs_m, df["macs_m"].to_numpy(), df["accuracy"].to_numpy()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained FEAST checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cinic10", "tinyimagenet"])
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data (default: ./data/<dataset>)")
    parser.add_argument("--arch_json", type=str, default=None,
                         help="A single search_single_subnet.py output JSON (d/e/w_indices)")
    parser.add_argument("--arch_json_dir", type=str, default=None,
                         help="Directory of search_single_subnet.py output JSONs; every *.json in it is evaluated")
    parser.add_argument("--reference_curve", type=str, default=None,
                         help="Optional evaluate.py-style CSV (macs_m, accuracy) to compute the gap against, "
                              "via linear interpolation at each found architecture's realized MAC value")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bn_calibration_split", type=float, default=0.1)
    parser.add_argument("--augmentation", type=str, default="basic", choices=["basic", "mixaug"],
                         help="Must match the augmentation used during training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for results")
    args = parser.parse_args()

    if not args.arch_json and not args.arch_json_dir:
        parser.error("Pass --arch_json or --arch_json_dir")

    if args.data_dir is None:
        args.data_dir = f"./data/{args.dataset}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "arch_params" not in checkpoint:
        raise ValueError("Checkpoint missing 'arch_params'. Cannot reconstruct model.")

    arch_params = checkpoint["arch_params"]
    if arch_params.get("resource_heterogeneity", False):
        arch_params = dict(arch_params)
        arch_params["resource_heterogeneity"] = False

    server_model = GenericServerOFA(
        arch_params=arch_params,
        sampling_method="TS_optimal_path",
        num_cli_total=1,
        bn_gamma_zero_init=arch_params.get("bn_gamma_zero_init", False),
    )
    server_model.set_model_params(checkpoint["params"])
    model = server_model.model.to(device)
    logger.info("Model loaded successfully.")

    test_loader = load_test_data(args.data_dir, args.batch_size, dataset=args.dataset)
    bn_loader, _ = load_bn_calibration_data(
        args.data_dir, batch_size=64, bn_calibration_split=args.bn_calibration_split,
        dataset=args.dataset, augmentation=args.augmentation,
    )

    if args.arch_json:
        json_paths = [args.arch_json]
    else:
        json_paths = sorted(glob.glob(os.path.join(args.arch_json_dir, "*.json")))
        if not json_paths:
            raise FileNotFoundError(f"No *.json files found in {args.arch_json_dir}")

    results = []
    for json_path in json_paths:
        arch, raw = load_arch_json(json_path)
        e_indices = e_values_to_indices(arch["e"], arch_params["expansion_ratio_choices"])

        model.set_active_subnet(d=arch["d"], e_indices=e_indices, w_indices=arch["w_indices"])

        if next(model.parameters()).device.type == "cpu" and torch.cuda.is_available():
            model = model.cuda()

        _reseed_bn_cal()
        set_running_statistics(model, bn_loader)

        acc, loss = evaluate_subnet(model, test_loader, device)

        macs, params = subnet_macs(
            depth_vec=arch["d"], exp_vec=arch["e"], w_indices=arch["w_indices"],
            width_mult_options=arch_params["width_multiplier_choices"], arch_config_params=arch_params,
        )
        macs_m = macs / 1e6

        row = {
            "arch_json": os.path.basename(json_path),
            "target_macs_m": raw.get("target_macs", macs) / 1e6,
            "macs_m": macs_m,
            "params_m": params / 1e6,
            "accuracy": acc,
            "loss": loss,
            "d": str(arch["d"]),
            "e": str(arch["e"]),
            "w_indices": str(arch["w_indices"]),
        }

        if args.reference_curve:
            ref_acc = interpolate_reference_accuracy(args.reference_curve, macs_m)
            row["reference_accuracy"] = ref_acc
            row["gap_pp"] = (ref_acc - acc) * 100

        results.append(row)
        logger.info(
            f"{os.path.basename(json_path):40s} | MACs: {macs_m:8.2f}M | Acc: {acc*100:6.2f}%"
            + (f" | Ref: {row['reference_accuracy']*100:6.2f}% | Gap: {row['gap_pp']:+.2f}pp"
               if args.reference_curve else "")
        )

    results_df = pd.DataFrame(results)

    if args.output:
        output_path = args.output
    else:
        base = args.arch_json_dir if args.arch_json_dir else os.path.dirname(args.arch_json)
        output_path = os.path.join(base or ".", "posthoc_evaluation_results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    logger.info(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    logger.info(f"Architectures evaluated: {len(results_df)}")
    logger.info(f"Mean accuracy: {results_df['accuracy'].mean()*100:.2f}%")
    if args.reference_curve:
        logger.info(f"Mean |gap|: {results_df['gap_pp'].abs().mean():.2f}pp")


if __name__ == "__main__":
    main()
