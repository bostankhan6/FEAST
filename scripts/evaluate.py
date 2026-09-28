#!/usr/bin/env python
"""
evaluate.py - Final Subnet Evaluation on the official test set

This script loads a trained model checkpoint and evaluates specified subnets
on the OFFICIAL test set. This test set is NEVER seen
during training, ensuring fair comparison across different methods.

Supports: CIFAR-100, CINIC-10, TinyImageNet

Usage:
    python evaluate.py --checkpoint path/to/model.pt --dataset cifar100

    # Evaluate specific subnets by index:
    python evaluate.py --checkpoint path/to/model.pt --subnet_indices 0 1 2 3
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

from feast.data.cifar100.data_loader import _data_transforms_cifar100, _data_transforms_cifar100_mixaug
from feast.data.cifar100.datasets import CIFAR100_truncated
from feast.data.cinic10.data_loader import _data_transforms_cinic10, _data_transforms_cinic10_mixaug
from feast.data.cinic10.datasets import ImageFolderTruncated
from feast.data.tinyimagenet.data_loader import _data_transforms_tinyimagenet, _data_transforms_tinyimagenet_mixaug
from feast.Server.generic_server_model import GenericServerOFA
from feast.utils.subnet_cost import subnet_macs
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BN-calibration is re-run once per evaluated subnet, all sharing one
# bn_loader. Without a per-call reseed, each pass consumes wherever the
# global RNG stream happens to be, so the same subnet gives a different
# result depending on how many other subnets were evaluated earlier in the
# same run (e.g. a 5-anchor run vs the full 56-point curve run disagreeing
# at shared MAC points). Reseeding to a fixed value immediately before every
# BN-cal call makes each subnet's calibration bit-identical regardless of
# run composition or order. See the matching fix in evaluate_compare.py.
BN_CAL_SEED = 0


def _reseed_bn_cal(seed=BN_CAL_SEED):
    import random as _random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)


def load_test_data(data_dir, batch_size=256, dataset='cifar100'):
    """Load the official CIFAR-100/CINIC-10/TinyImageNet test set."""
    if dataset == 'cifar100':
        _, transform_test = _data_transforms_cifar100()
        test_ds = CIFAR100_truncated(data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'cinic10':
        _, transform_test = _data_transforms_cinic10()
        test_dir = os.path.join(data_dir, 'test')
        test_ds = ImageFolderTruncated(test_dir, transform=transform_test)
    elif dataset == 'tinyimagenet':
        _, transform_test = _data_transforms_tinyimagenet()
        test_dir = os.path.join(data_dir, 'val')
        test_ds = ImageFolderTruncated(test_dir, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    logger.info(f"Loaded official {dataset.upper()} test set: {len(test_ds)} samples")
    return test_loader


def load_bn_calibration_data(data_dir, batch_size=64, bn_calibration_split=0.1, dataset='cifar100', augmentation='basic'):
    """
    Load BN calibration subset from validation set WITH TRAINING AUGMENTATIONS.

    Uses the SAME fixed seed and split logic as training to ensure consistency.
    Applies training augmentations (RandomCrop, HorizontalFlip, etc.) for BN calibration.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for BN calibration
        bn_calibration_split: Fraction of validation set to use for calibration
        dataset: Dataset name (cifar100, cinic10, tinyimagenet)
        augmentation: 'basic' or 'mixaug' — must match the augmentation used during training

    Returns:
        bn_loader: DataLoader for BN calibration (WITH training augmentations)
        eval_size: Size of remaining evaluation set (for reference)
    """
    if dataset == 'cifar100':
        if augmentation == 'mixaug':
            transform_train, _ = _data_transforms_cifar100_mixaug()
        else:
            transform_train, _ = _data_transforms_cifar100()
        full_train = CIFAR100_truncated(data_dir, train=True, download=True, transform=transform_train)
        validation_split = 0.1

    elif dataset == 'cinic10':
        # CINIC-10 has explicit valid folder
        if augmentation == 'mixaug':
            train_transform, _ = _data_transforms_cinic10_mixaug()
        else:
            train_transform, _ = _data_transforms_cinic10()

        valid_dir = os.path.join(data_dir, 'valid')
        full_val_ds = ImageFolderTruncated(valid_dir, transform=train_transform)

        total_samples = len(full_val_ds)
        bn_cal_size = int(total_samples * bn_calibration_split)
        bn_cal_size = max(bn_cal_size, batch_size)

        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_samples, generator=g).tolist()
        bn_cal_indices = indices[:bn_cal_size]

        bn_subset = torch.utils.data.Subset(full_val_ds, bn_cal_indices)
        bn_loader = torch.utils.data.DataLoader(
            bn_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        eval_size = total_samples - bn_cal_size
        logger.info(f"Loaded BN calibration set: {len(bn_subset)} samples (WITH training augmentations)")
        return bn_loader, eval_size
    elif dataset == 'tinyimagenet':
        if augmentation == 'mixaug':
            train_transform, _ = _data_transforms_tinyimagenet_mixaug()
        else:
            train_transform, _ = _data_transforms_tinyimagenet()
        train_dir = os.path.join(data_dir, 'train')
        full_train = ImageFolderTruncated(train_dir, transform=train_transform)
        total_samples = len(full_train)

        np.random.seed(42)
        all_indices = np.random.permutation(total_samples)
        n_val = int(total_samples * 0.1)
        val_indices = all_indices[:n_val]

        bn_cal_size = int(n_val * bn_calibration_split)
        bn_cal_size = max(bn_cal_size, batch_size)

        g = torch.Generator().manual_seed(42)
        val_perm = torch.randperm(n_val, generator=g).tolist()
        bn_cal_indices = [val_indices[i] for i in val_perm[:bn_cal_size]]

        bn_subset = torch.utils.data.Subset(full_train, bn_cal_indices)
        bn_loader = torch.utils.data.DataLoader(
            bn_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        eval_size = n_val - bn_cal_size
        logger.info(f"Loaded BN calibration set: {len(bn_subset)} samples (WITH training augmentations)")
        return bn_loader, eval_size
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # For CIFAR-100: Use validation split from training data
    total_samples = len(full_train)
    
    # First split: training vs validation (same seed as partition_data)
    np.random.seed(42)
    all_indices = np.random.permutation(total_samples)
    n_val = int(total_samples * validation_split)
    val_indices = all_indices[:n_val]
    
    # Second split: calibration vs evaluation (same seed as data_loader)
    bn_cal_size = int(n_val * bn_calibration_split)
    bn_cal_size = max(bn_cal_size, batch_size)
    
    g = torch.Generator().manual_seed(42)
    val_perm = torch.randperm(n_val, generator=g).tolist()
    bn_cal_indices = [val_indices[i] for i in val_perm[:bn_cal_size]]
    
    bn_subset = torch.utils.data.Subset(full_train, bn_cal_indices)
    # shuffle=False: bn_cal_indices are already randomly selected via the
    # seed=42 permutation above; a shuffling sampler holds a persistent
    # generator whose state advances every time bn_loader is iterated, so
    # sample order (and hence BN running stats) would otherwise depend on
    # how many prior subnets were evaluated in the same run. Fixed order
    # removes that dependency. num_workers=4 for eval speed: verified
    # bit-identical across reruns at this setting (canonical numbers were
    # (re-)established under num_workers=4 uniformly across all datasets).
    bn_loader = torch.utils.data.DataLoader(
        bn_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    eval_size = n_val - bn_cal_size
    logger.info(f"Loaded BN calibration set: {len(bn_subset)} samples (WITH training augmentations)")
    return bn_loader, eval_size


def evaluate_subnet(model, test_loader, device):
    """Evaluate a subnet on the test set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Final Subnet Evaluation on CIFAR-100/CINIC-10/TinyImageNet Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cinic10", "tinyimagenet"], help="Dataset to evaluate on")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data (default: ./data/<dataset>)")
    parser.add_argument("--subnet_cache", type=str, default=None, help="Path to subnet cache CSV")
    parser.add_argument("--subnet_indices", type=int, nargs="+", default=None, help="Specific subnet indices to evaluate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--bn_calibration_split", type=float, default=0.1, help="Fraction of validation set for BN calibration (must match training)")
    parser.add_argument("--augmentation", type=str, default="basic", choices=["basic", "mixaug"],
                        help="Augmentation used during training — selects matching BN calibration transform")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for results")
    args = parser.parse_args()

    # Set default data_dir based on dataset
    if args.data_dir is None:
        args.data_dir = f"./data/{args.dataset}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset: {args.dataset.upper()}")

    # --- Load Checkpoint ---
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if "arch_params" not in checkpoint:
        raise ValueError("Checkpoint missing 'arch_params'. Cannot reconstruct model.")
    
    arch_params = checkpoint["arch_params"]
    logger.info(f"Loaded architecture parameters: num_stages={arch_params['num_stages']}, n_classes={arch_params['n_classes']}")

    # Eval doesn't need client budgets; disable resource heterogeneity to avoid
    # numpy.random.zipf(a<=1) crashes for ckpts trained with low Zipf alphas.
    if arch_params.get('resource_heterogeneity', False):
        arch_params = dict(arch_params)
        arch_params['resource_heterogeneity'] = False

    # --- Create Model ---
    server_model = GenericServerOFA(
        arch_params=arch_params,
        sampling_method="TS_optimal_path",
        num_cli_total=1,  # Not used for evaluation
        bn_gamma_zero_init=arch_params.get('bn_gamma_zero_init', False),
    )
    server_model.set_model_params(checkpoint["params"])
    model = server_model.model.to(device)
    logger.info("Model loaded successfully.")

    # --- Load Data ---
    test_loader = load_test_data(args.data_dir, args.batch_size, dataset=args.dataset)
    bn_loader, _ = load_bn_calibration_data(args.data_dir, batch_size=64, bn_calibration_split=args.bn_calibration_split, dataset=args.dataset, augmentation=args.augmentation)

    # --- Determine Subnets to Evaluate ---
    subnets_to_eval = []
    
    if args.subnet_cache:
        logger.info(f"Loading subnet cache: {args.subnet_cache}")
        df = pd.read_csv(args.subnet_cache)
        df = df.sort_values(by='macs').reset_index(drop=True)
        
        import ast
        for idx, row in df.iterrows():
            if args.subnet_indices is None or idx in args.subnet_indices:
                subnet = {
                    'd': ast.literal_eval(row['d']),
                    'e': ast.literal_eval(row['e']),
                    'w_indices': ast.literal_eval(row['w_indices']),
                }
                subnets_to_eval.append((idx, subnet, row['macs']))
    else:
        # Default: evaluate min and max subnets
        logger.info("No subnet cache provided. Evaluating min and max subnets only.")
        min_arch = server_model.min_subnet_arch()
        max_arch = server_model.max_subnet_arch()
        subnets_to_eval = [
            (0, min_arch, server_model._macs_min),
            (1, max_arch, server_model._macs_max),
        ]

    # --- Evaluate Each Subnet ---
    results = []
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL EVALUATION ON OFFICIAL {args.dataset.upper()} TEST SET")
    logger.info(f"{'='*60}")
    
    for idx, subnet, macs in tqdm(subnets_to_eval, desc="Evaluating subnets"):
        # Configure model for this subnet
        # Get e values and convert to indices if needed
        e_values = subnet['e']
        if isinstance(e_values[0], float):
            # e contains actual expansion ratios, convert to indices
            e_indices = [arch_params['expansion_ratio_choices'].index(e) for e in e_values]
        else:
            # e contains indices already
            e_indices = e_values
        
        model.set_active_subnet(
            d=subnet['d'],
            e_indices=e_indices,
            w_indices=subnet['w_indices']
        )
        
        # Recalibrate BN statistics
        import time
        bn_start = time.time()
        
        # Ensure model is on correct device before BN Reset
        if next(model.parameters()).device.type == 'cpu' and torch.cuda.is_available():
            logger.warning("[BN Reset] Model is on CPU! Moving to GPU...")
            model = model.cuda()

        _reseed_bn_cal()
        set_running_statistics(model, bn_loader)
        
        bn_elapsed = time.time() - bn_start
        logger.info(f"BN Reset time: {bn_elapsed:.2f}s")
        
        # Evaluate
        acc, loss = evaluate_subnet(model, test_loader, device)
        
        # Recalculate MACs for consistency
        recalc_macs, _ = subnet_macs(
            depth_vec=subnet['d'],
            exp_vec=e_values,
            w_indices=subnet['w_indices'],
            width_mult_options=arch_params['width_multiplier_choices'],
            arch_config_params=arch_params
        )
        
        results.append({
            'subnet_idx': idx,
            'macs_m': recalc_macs / 1e6,
            'accuracy': acc,
            'loss': loss,
            'd': str(subnet['d']),
            'e': str(subnet['e']),
            'w_indices': str(subnet['w_indices']),
        })
        
        logger.info(f"Subnet {idx:2d} | MACs: {recalc_macs/1e6:8.2f}M | Acc: {acc*100:6.2f}% | Loss: {loss:.4f}")

    # --- Save Results ---
    results_df = pd.DataFrame(results)
    
    if args.output:
        output_path = args.output
    else:
        # Default: save next to checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(ckpt_dir, "final_evaluation_results.csv")
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")
    
    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total subnets evaluated: {len(results)}")
    logger.info(f"Best accuracy: {results_df['accuracy'].max()*100:.2f}% (Subnet {results_df.loc[results_df['accuracy'].idxmax(), 'subnet_idx']})")
    logger.info(f"Average accuracy: {results_df['accuracy'].mean()*100:.2f}%")
    

if __name__ == "__main__":
    main()
