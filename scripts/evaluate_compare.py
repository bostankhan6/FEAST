#!/usr/bin/env python
"""
Unified evaluation comparing FEAST (full method) vs HeteroFL vs ScaleFL vs FIARSE.

All methods evaluated on:
  - Official CIFAR-100 test set (10K samples)
  - With BN calibration from the same fixed validation split (500 samples)
    using training augmentations — matching the protocol in scripts/evaluate.py

Usage:
    ./venv/bin/python3 scripts/evaluate_compare.py \
        --feast_checkpoint  checkpoints/feast-cifar100-mixaug/best_checkpoint_supernet.pt \
        --heterofl_checkpoint checkpoints/heterofl/best_full.pt \
        --scalefl_checkpoint  checkpoints/scalefl/best_full.pt \
        --fiarse_checkpoint   checkpoints/fiarse/best_full.pt \
        --data_dir data/cifar100 \
        --gpu 0
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

from feast.data.cifar100.data_loader import _data_transforms_cifar100, _data_transforms_cifar100_mixaug
from feast.data.cifar100.datasets import CIFAR100_truncated
from feast.data.cinic10.datasets import ImageFolderTruncated
from feast.data.cinic10.data_loader import _data_transforms_cinic10, _data_transforms_cinic10_mixaug
from feast.data.tinyimagenet.data_loader import _data_transforms_tinyimagenet, _data_transforms_tinyimagenet_mixaug
from feast.Server.generic_server_model import GenericServerOFA
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

from baselines.heterofl.model import build_global_model as heterofl_build_global
from baselines.heterofl.federation import HeteroFLFederation, TIER_RATES, TIER_MACS
from baselines.heterofl.trainer import evaluate_tier

from baselines.scalefl.model import build_global_model as scalefl_build_global
from baselines.scalefl.federation import ScaleFLFederation
from baselines.scalefl.split_config import get_default_configs
from baselines.scalefl.trainer import evaluate_level

from baselines.fiarse.model import build_global_model as fiarse_build_global, FULL_MODEL_MACS
from baselines.fiarse.trainer import evaluate_at_model_size

FIARSE_EVAL_MODEL_SIZES = [0.015625, 0.0625, 0.125, 0.25, 0.5, 1.0]

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# BN-calibration passes are re-run once per variant/tier/level/size, all sharing
# one bn_loader. Without a per-call reseed, each pass consumes wherever the
# global RNG stream happens to be, so the same operating point (e.g. 25M)
# gives a different result depending on how many other variants were
# evaluated earlier in the same run (5-anchor run vs 56-point curve run).
# Reseeding to a fixed value immediately before every BN-cal call makes each
# variant's calibration bit-identical regardless of run composition or order.
BN_CAL_SEED = 0


def _reseed_bn_cal(seed=BN_CAL_SEED):
    import random as _random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)


# ---------------------------------------------------------------------------
# FEAST subnet specs (identical configs across datasets; MACs differ by dataset)
# ---------------------------------------------------------------------------

# CIFAR-100 / CINIC-10 (32×32 input, stem_stride=1): 25M / 50M / 229M / 408M / 596M
_FEAST_SUBNET_CONFIGS = [
    {"d": [3, 7, 6, 7], "e": [0.18, 0.18, 0.1, 0.1],  "w_indices": [2, 0, 0, 0, 0]},
    {"d": [5, 6, 7, 8], "e": [0.18, 0.1, 0.14, 0.22], "w_indices": [2, 0, 1, 1, 0]},
    {"d": [8, 8, 8, 8], "e": [0.22, 0.1, 0.1, 0.1],   "w_indices": [4, 2, 3, 4, 3]},
    {"d": [8, 8, 8, 8], "e": [0.1, 0.1, 0.14, 0.1],   "w_indices": [6, 6, 6, 3, 5]},
    {"d": [8, 8, 8, 8], "e": [0.18, 0.1, 0.1, 0.1],   "w_indices": [9, 4, 6, 7, 6]},
]

_FEAST_SUBNET_LABELS = {
    "cifar100":      ["0 (~25M)", "1 (~50M)", "2 (~229M)", "3 (~408M)", "4 (~596M)"],
    "cinic10":       ["0 (~25M)", "1 (~50M)", "2 (~229M)", "3 (~408M)", "4 (~596M)"],
    # TinyImageNet: same architectures, MACs shift slightly with stem_stride=2 at 64×64.
    # Post-stem feature map stays 32×32 so body MACs are ~identical; stem adds ~same.
    # Actual values (from extended_range_tinyimagenet cache): 27M/52M/232M/412M/601M.
    "tinyimagenet":  ["0 (~27M)", "1 (~52M)", "2 (~232M)", "3 (~412M)", "4 (~601M)"],
}

FEAST_SUBNETS = {
    "0 (~25M)":  _FEAST_SUBNET_CONFIGS[0],
    "1 (~50M)":  _FEAST_SUBNET_CONFIGS[1],
    "2 (~229M)": _FEAST_SUBNET_CONFIGS[2],
    "3 (~408M)": _FEAST_SUBNET_CONFIGS[3],
    "4 (~596M)": _FEAST_SUBNET_CONFIGS[4],
}


def get_feast_subnets(dataset: str) -> dict:
    """Return FEAST subnet spec dict with dataset-appropriate MAC labels."""
    labels = _FEAST_SUBNET_LABELS.get(dataset, _FEAST_SUBNET_LABELS["cifar100"])
    return {label: cfg for label, cfg in zip(labels, _FEAST_SUBNET_CONFIGS)}


# ---------------------------------------------------------------------------
# Data loaders (identical to scripts/evaluate.py)
# ---------------------------------------------------------------------------

def load_test_loader(data_dir, batch_size=256, dataset='cifar100'):
    if dataset == 'cifar100':
        _, transform_test = _data_transforms_cifar100()
        test_ds = CIFAR100_truncated(data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'tinyimagenet':
        _, transform_test = _data_transforms_tinyimagenet()
        # val/ is the held-out test set (10k images, 200 classes)
        test_ds = ImageFolderTruncated(os.path.join(data_dir, 'val'), transform=transform_test)
    else:  # cinic10
        _, transform_test = _data_transforms_cinic10()
        test_ds = ImageFolderTruncated(os.path.join(data_dir, 'test'), transform=transform_test)
    # Shuffling is required for baselines that use track_running_stats=False BN:
    # ImageFolder loads in class-alphabetical order, producing single-class batches that
    # give unrepresentative BN statistics at inference time.  Shuffling ensures each batch
    # contains a mix of classes, matching the random-index validation protocol used during
    # training.  FEAST is unaffected (it uses calibrated running stats via set_running_statistics).
    #
    # This loader is iterated once per variant/tier/level/size across every method's
    # loop in main(), all against the same loader object. shuffle=True with a generator
    # holds persistent sampler state that advances on every __iter__ call, so batch
    # composition (and hence accuracy for track_running_stats=False baselines) would
    # depend on how many other variants were evaluated earlier in the same run.
    # Precomputing the shuffle once and iterating in fixed order removes that dependency.
    g = torch.Generator().manual_seed(0)
    fixed_order = torch.randperm(len(test_ds), generator=g).tolist()
    test_ds = torch.utils.data.Subset(test_ds, fixed_order)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Test set: {len(test_ds)} samples ({dataset})")
    return loader


def load_bn_calibration_loader(data_dir, batch_size=64, dataset='cifar100', augmentation='basic'):
    """
    CIFAR-100:    10% of 50K train → 5K val, then 10% of that → 500 BN cal samples.
    CINIC-10:     10% of 90K pre-built valid set → 9K BN cal samples, seed=42.
    TinyImageNet: 10% of 100K train → 10K val (seed=42), then 10% of that → 1K BN cal.
                  Replicates load_partition_data_tinyimagenet(validation_split=0.1,
                  bn_calibration_split=0.1) exactly.

    augmentation: 'basic' or 'mixaug' — must match the augmentation used during training.
    """
    if dataset == 'cifar100':
        if augmentation == 'mixaug':
            transform_train, _ = _data_transforms_cifar100_mixaug()
        else:
            transform_train, _ = _data_transforms_cifar100()
        full_train = CIFAR100_truncated(data_dir, train=True, download=True,
                                        transform=transform_train)
        total = len(full_train)
        np.random.seed(42)
        all_idx = np.random.permutation(total)
        n_val = int(total * 0.1)
        val_idx = all_idx[:n_val]

        bn_cal_size = max(int(n_val * 0.1), batch_size)
        g = torch.Generator().manual_seed(42)
        val_perm = torch.randperm(n_val, generator=g).tolist()
        bn_cal_idx = [val_idx[i] for i in val_perm[:bn_cal_size]]

        bn_subset = torch.utils.data.Subset(full_train, bn_cal_idx)
    elif dataset == 'tinyimagenet':
        if augmentation == 'mixaug':
            transform_train, _ = _data_transforms_tinyimagenet_mixaug()
        else:
            transform_train, _ = _data_transforms_tinyimagenet()
        train_dir = os.path.join(data_dir, 'train')
        full_train = ImageFolderTruncated(train_dir, transform=transform_train)
        n_train_full = len(full_train)  # 100k

        # Replicate validation split from partition_data (seed=42, 10% held out)
        np.random.seed(42)
        all_indices = np.random.permutation(n_train_full)
        n_val = int(n_train_full * 0.1)  # 10000
        val_indices = all_indices[:n_val]

        # Replicate BN cal split (seed=42, 10% of val → 1000 samples)
        bn_cal_size = max(int(n_val * 0.1), batch_size)  # 1000
        g = torch.Generator().manual_seed(42)
        val_perm = torch.randperm(n_val, generator=g).tolist()
        bn_cal_global_idx = val_indices[val_perm[:bn_cal_size]]

        bn_subset = torch.utils.data.Subset(full_train, bn_cal_global_idx.tolist())
    else:  # cinic10
        if augmentation == 'mixaug':
            transform_train, _ = _data_transforms_cinic10_mixaug()
        else:
            transform_train, _ = _data_transforms_cinic10()
        full_val = ImageFolderTruncated(os.path.join(data_dir, 'valid'),
                                        transform=transform_train)
        total = len(full_val)
        bn_cal_size = max(int(total * 0.1), batch_size)
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(total, generator=g).tolist()
        bn_subset = torch.utils.data.Subset(full_val, indices[:bn_cal_size])

    # shuffle=False: bn_subset indices are already randomly selected via the
    # seed=42/seed=0 permutations above. A shuffling sampler holds a
    # persistent generator whose state advances every time bn_loader is
    # iterated, so sample order (and hence BN running stats) would otherwise
    # depend on how many prior variants/tiers/levels were evaluated in the
    # same run. Fixed order removes that dependency. num_workers=4 for eval
    # speed: verified bit-identical across reruns at this setting (canonical
    # numbers were (re-)established under num_workers=4 uniformly across all
    # datasets).
    loader = torch.utils.data.DataLoader(bn_subset, batch_size=batch_size,
                                          shuffle=False, num_workers=4,
                                          pin_memory=True)
    logger.info(f"BN calibration: {len(bn_subset)} samples (with training augmentation)")
    return loader


# ---------------------------------------------------------------------------
# FEAST evaluation
# ---------------------------------------------------------------------------

def evaluate_feast(checkpoint_path, test_loader, bn_loader, device, dataset='cifar100'):
    logger.info("=== Evaluating FEAST (Full Method) ===")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch_params = ckpt["arch_params"]
    exp_choices = arch_params["expansion_ratio_choices"]

    if arch_params.get('resource_heterogeneity', False):
        # Resource-constrained client-budget assignment is irrelevant at eval
        # time (single-supernet load, no federation) and np.random.zipf
        # requires alpha > 1 — checkpoints trained with zipf_alpha <= 1.0
        # (e.g. the s=0.8/1.0 sensitivity sweep) would otherwise crash here.
        arch_params = dict(arch_params)
        arch_params['resource_heterogeneity'] = False

    server_model = GenericServerOFA(
        arch_params=arch_params,
        sampling_method="TS_optimal_path",
        num_cli_total=1,
        bn_gamma_zero_init=arch_params.get("bn_gamma_zero_init", False),
    )
    server_model.set_model_params(ckpt["params"])
    supernet = server_model.model.to(device)

    subnets = get_feast_subnets(dataset)
    results = {}
    for label, subnet in subnets.items():
        # Reset params to clean state for each subnet's BN calibration
        server_model.set_model_params(ckpt["params"])
        supernet = server_model.model.to(device)
        supernet.eval()

        e_indices = [exp_choices.index(v) for v in subnet["e"]]
        supernet.set_active_subnet(
            d=subnet["d"], e_indices=e_indices, w_indices=subnet["w_indices"]
        )
        _reseed_bn_cal()
        set_running_statistics(supernet, bn_loader)

        supernet.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = supernet(x).argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        results[label] = acc
        logger.info(f"  FEAST subnet {label}: {acc:.2f}%")

    logger.info(f"  Trained for {ckpt.get('round', '?')} rounds")
    return results


# ---------------------------------------------------------------------------
# HeteroFL evaluation
# ---------------------------------------------------------------------------

def evaluate_heterofl(checkpoint_path, test_loader, bn_loader, device, num_classes=100,
                      stem_stride=1):
    logger.info("=== Evaluating HeteroFL ===")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    global_model = heterofl_build_global(num_classes=num_classes, stem_stride=stem_stride)
    global_model.load_state_dict(ckpt["model_state_dict"])
    global_model.to(device)

    federation = HeteroFLFederation(global_model)

    results = {}
    for tier in ["e", "d", "c", "b", "a"]:
        macs_m = TIER_MACS[tier] / 1e6
        _reseed_bn_cal()
        acc = evaluate_tier(tier, global_model, federation,
                            test_loader, device, bn_calibration_loader=bn_loader,
                            num_classes=num_classes,
                            stem_stride=stem_stride)
        label = f"{tier} (~{macs_m:.0f}M)"
        results[label] = acc
        logger.info(f"  HeteroFL tier {label}: {acc:.2f}%")

    logger.info(f"  Trained for {ckpt.get('round', '?')} rounds")
    return results


# ---------------------------------------------------------------------------
# ScaleFL evaluation
# ---------------------------------------------------------------------------

def evaluate_scalefl(checkpoint_path, test_loader, bn_loader, device, num_classes=100,
                     stem_stride=1):
    logger.info("=== Evaluating ScaleFL ===")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    configs, _ = get_default_configs()
    global_model = scalefl_build_global(num_classes=num_classes, stem_stride=stem_stride)
    global_model.load_state_dict(ckpt["model_state_dict"])
    global_model.to(device)

    federation = ScaleFLFederation(global_model)

    results = {}
    for cfg in configs:
        level = cfg["level"]
        macs_m = cfg["macs"] / 1e6
        _reseed_bn_cal()
        acc = evaluate_level(level, global_model, federation,
                             test_loader, device, bn_calibration_loader=bn_loader,
                             num_classes=num_classes,
                             stem_stride=stem_stride)
        label = f"{level} (~{macs_m:.0f}M)"
        results[label] = acc
        logger.info(f"  ScaleFL level {label}: {acc:.2f}%")

    logger.info(f"  Trained for {ckpt.get('round', '?')} rounds")
    return results


# ---------------------------------------------------------------------------
# FIARSE evaluation
# ---------------------------------------------------------------------------

def evaluate_fiarse(checkpoint_path, test_loader, bn_loader, device, num_classes=100,
                    stem_stride=1):
    logger.info("=== Evaluating FIARSE ===")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    global_model = fiarse_build_global(num_classes=num_classes, stem_stride=stem_stride)
    global_model.load_state_dict(ckpt["model_state_dict"])
    global_model.to(device)

    results = {}
    for ms in FIARSE_EVAL_MODEL_SIZES:
        macs_m = ms * FULL_MODEL_MACS / 1e6
        _reseed_bn_cal()
        acc = evaluate_at_model_size(global_model, ms, test_loader, device,
                                     bn_calibration_loader=bn_loader)
        label = f"ms={ms:.4f} (~{macs_m:.0f}M)"
        results[label] = acc
        logger.info(f"  FIARSE {label}: {acc:.2f}%")

    logger.info(f"  Trained for {ckpt.get('round', '?')} rounds")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feast_checkpoint", type=str, default=None)
    parser.add_argument("--heterofl_checkpoint", type=str, default=None)
    parser.add_argument("--scalefl_checkpoint", type=str, default=None)
    parser.add_argument("--fiarse_checkpoint", type=str,
                        default=None)
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cinic10", "tinyimagenet"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory. Defaults to data/<dataset>.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--augmentation", type=str, default="basic", choices=["basic", "mixaug"],
                        help="Augmentation used during training — selects matching BN calibration transform")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"

    if args.dataset == "cifar100":
        num_classes = 100
        stem_stride = 1
    elif args.dataset == "cinic10":
        num_classes = 10
        stem_stride = 1
    else:  # tinyimagenet
        num_classes = 200
        stem_stride = 2

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Dataset: {args.dataset} | "
                f"num_classes: {num_classes} | stem_stride: {stem_stride}")

    # Deterministic BN calibration: the mixaug/basic augmentation forward passes
    # consume the global RNG, so without a fixed seed the recomputed running stats
    # (and thus reported accuracies) drift ~0.1-0.3pp run-to-run. Seed here so the
    # eval is reproducible and re-runs reproduce the canonical numbers.
    import random as _random
    torch.manual_seed(0); np.random.seed(0); _random.seed(0)
    torch.cuda.manual_seed_all(0)

    test_loader = load_test_loader(args.data_dir, args.batch_size, dataset=args.dataset)
    bn_loader   = load_bn_calibration_loader(args.data_dir, dataset=args.dataset, augmentation=args.augmentation)

    # All methods: skip gracefully if no checkpoint provided or file missing
    def _ckpt_exists(path):
        return path is not None and os.path.isfile(path)

    feast_results = evaluate_feast(args.feast_checkpoint, test_loader, bn_loader, device,
                                    dataset=args.dataset) if _ckpt_exists(args.feast_checkpoint) else None

    heterofl_results = evaluate_heterofl(
        args.heterofl_checkpoint, test_loader, bn_loader, device,
        num_classes=num_classes, stem_stride=stem_stride,
    ) if _ckpt_exists(args.heterofl_checkpoint) else None

    scalefl_results = evaluate_scalefl(
        args.scalefl_checkpoint, test_loader, bn_loader, device,
        num_classes=num_classes, stem_stride=stem_stride,
    ) if _ckpt_exists(args.scalefl_checkpoint) else None

    fiarse_results = evaluate_fiarse(
        args.fiarse_checkpoint, test_loader, bn_loader, device,
        num_classes=num_classes, stem_stride=stem_stride,
    ) if _ckpt_exists(args.fiarse_checkpoint) else None

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    W = 60
    dataset_labels = {"cifar100": "CIFAR-100", "cinic10": "CINIC-10",
                      "tinyimagenet": "TinyImageNet-200"}
    dataset_label = dataset_labels.get(args.dataset, args.dataset)
    test_n = len(test_loader.dataset)
    print("\n" + "=" * W)
    print(" COMPARISON: FEAST vs baselines")
    print(f"  Dataset: {dataset_label} official test set ({test_n:,} samples)")
    print("  BN calibration: validation samples with training augmentation")
    print("=" * W)

    if feast_results is not None:
        print("\n--- FEAST (per-step min/random/max training + sub-supernet, γ=1, OFA supernet) ---")
        print(f"  {'Subnet':<22} {'Test Acc':>9}")
        print(f"  {'-'*32}")
        for label, acc in feast_results.items():
            print(f"  {'Subnet ' + label:<22} {acc:>8.2f}%")
        feast_vals = list(feast_results.values())
        print(f"  {'Min → Max':<22} {min(feast_vals):>7.2f}% → {max(feast_vals):.2f}%  (gap: {max(feast_vals)-min(feast_vals):+.2f}pp)")
    else:
        feast_vals = None

    if heterofl_results is not None:
        print("\n--- HeteroFL (width-only, ResNet) ---")
        print(f"  {'Tier':<22} {'Test Acc':>9}")
        print(f"  {'-'*32}")
        for label, acc in heterofl_results.items():
            print(f"  {'Tier ' + label:<22} {acc:>8.2f}%")
        hfl_vals = list(heterofl_results.values())
        print(f"  {'Min → Max':<22} {min(hfl_vals):>7.2f}% → {max(hfl_vals):.2f}%  (gap: {max(hfl_vals)-min(hfl_vals):+.2f}pp)")
    else:
        hfl_vals = None

    if scalefl_results is not None:
        print("\n--- ScaleFL (2D split + self-distil, ResNet) ---")
        print(f"  {'Level':<22} {'Test Acc':>9}")
        print(f"  {'-'*32}")
        for label, acc in scalefl_results.items():
            print(f"  {'Level ' + label:<22} {acc:>8.2f}%")
        sfl_vals = list(scalefl_results.values())
        print(f"  {'Min → Max':<22} {min(sfl_vals):>7.2f}% → {max(sfl_vals):.2f}%  (gap: {max(sfl_vals)-min(sfl_vals):+.2f}pp)")
    else:
        sfl_vals = None


    if fiarse_results is not None:
        print("\n--- FIARSE (unstructured pruning + TCB-GD, ResNet) ---")
        print(f"  {'Model Size':<26} {'Test Acc':>9}")
        print(f"  {'-'*36}")
        for label, acc in fiarse_results.items():
            print(f"  {label:<26} {acc:>8.2f}%")
        frs_vals = list(fiarse_results.values())
        print(f"  {'Min → Max':<26} {min(frs_vals):>7.2f}% → {max(frs_vals):.2f}%  (gap: {max(frs_vals)-min(frs_vals):+.2f}pp)")
    else:
        frs_vals = None

    print("\n" + "=" * W)
    print(" SUMMARY (Min / Max / Gap)")
    print("=" * W)
    print(f"  {'Method':<38} {'Min':>7} {'Max':>7} {'Gap':>8}")
    print(f"  {'-'*62}")
    if feast_vals is not None:
        print(f"  {'FEAST Full Method (OFA)':<38} {min(feast_vals):>6.2f}% {max(feast_vals):>6.2f}% {max(feast_vals)-min(feast_vals):>+7.2f}pp")
    if hfl_vals is not None:
        print(f"  {'HeteroFL (ResNet, 5 tiers)':<38} {min(hfl_vals):>6.2f}% {max(hfl_vals):>6.2f}% {max(hfl_vals)-min(hfl_vals):>+7.2f}pp")
    if sfl_vals is not None:
        print(f"  {'ScaleFL (ResNet, 4 levels)':<38} {min(sfl_vals):>6.2f}% {max(sfl_vals):>6.2f}% {max(sfl_vals)-min(sfl_vals):>+7.2f}pp")
    if frs_vals is not None:
        print(f"  {'FIARSE (ResNet, unstructured)':<38} {min(frs_vals):>6.2f}% {max(frs_vals):>6.2f}% {max(frs_vals)-min(frs_vals):>+7.2f}pp")
    print("=" * W)


if __name__ == "__main__":
    main()
