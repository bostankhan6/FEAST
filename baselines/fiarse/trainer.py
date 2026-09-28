"""
FIARSE Training Loop for CIFAR-100.

Faithful to FIARSE paper (Wu et al., NeurIPS 2024):
  - Clients receive masked model (TopK by |θ| at their model_size)
  - Local training: standard SGD, NO weight_decay, NO momentum (Table 3)
  - Bern TCB-GD in forward pass pushes params near threshold above/below
  - Client returns delta = (cache_params - trained_params)
  - Server aggregation: partial averaging of non-zero deltas,
    θ -= lr_global × avg_delta
  - BN: track_running_stats=False
  - lr_global = 1.0 (paper default)
  - Evaluation: TopK mask at various model_sizes, BN calibration

Data pipeline reuses feast load_partition_data_cifar100 for identical
data splits across all methods.
"""

import sys
import os
import copy
import logging
import argparse
import random
import gc

import torch
import torch.nn as nn
import wandb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import math
import numpy as _np

from feast.data.cifar100.data_loader import load_partition_data_cifar100
from feast.data.cinic10.data_loader import load_partition_data_cinic10
from feast.data.tinyimagenet.data_loader import load_partition_data_tinyimagenet

from .model import (build_global_model, budget_to_model_size,
                    FULL_MODEL_MACS, FIARSEResNet)
from .federation import FIARSEFederation
from baselines.utils import get_client_budgets
from baselines.bn_calibration import calibrate_bn_running_stats


CKPT_LATEST = 'checkpoint_latest.pt'
CKPT_BEST   = 'best_full.pt'


def _mixup_batch(x, y, alpha, device):
    lam = _np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return x * lam + x[idx] * (1 - lam), y, y[idx], lam


def _cutmix_batch(x, y, alpha, device):
    lam = _np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    _, _, H, W = x.shape
    cut_rat = (1 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cx = _np.random.randint(W)
    cy = _np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam


def _mix_batch(x, y, mode, mixup_alpha, cutmix_alpha, device):
    if mode == 'none':
        return x, y, y, 1.0
    if mode == 'alternating':
        mode = 'mixup' if _np.random.rand() < 0.5 else 'cutmix'
    if mode == 'mixup':
        return _mixup_batch(x, y, mixup_alpha, device)
    return _cutmix_batch(x, y, cutmix_alpha, device)


def _mixed_ce(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# Evaluation model sizes: roughly match HeteroFL tier MACs plus intermediates
EVAL_MODEL_SIZES = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.015625]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, round_idx, global_model, best_acc):
    torch.save({
        'round': round_idx,
        'model_state_dict': global_model.state_dict(),
        'best_full_acc': best_acc,
    }, path)


def load_checkpoint(path, global_model):
    ckpt = torch.load(path, map_location='cpu')
    global_model.load_state_dict(ckpt['model_state_dict'])
    return ckpt['round'], ckpt['best_full_acc']


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_one_client(client_model: FIARSEResNet,
                     data_loader,
                     lr: float,
                     local_epochs: int,
                     device: torch.device,
                     criterion: nn.Module = None,
                     mix_mode: str = 'none',
                     mixup_alpha: float = 0.4,
                     cutmix_alpha: float = 1.0):
    """
    FIARSE local training: standard SGD with Bern masking in forward pass.

    Following the paper (Table 3): NO weight_decay, NO momentum. The Bern
    TCB-GD function handles sparsity enforcement through its biased backward.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    cache_state = copy.deepcopy(client_model.state_dict())

    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)

    total_loss, n_batches = 0.0, 0
    for _ in range(local_epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            x_mix, y_a, y_b, lam = _mix_batch(x, y, mix_mode, mixup_alpha,
                                               cutmix_alpha, device)
            optimizer.zero_grad()
            logits = client_model(x_mix)
            loss = _mixed_ce(criterion, logits, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    trained_state = client_model.state_dict()
    delta = {}
    for name in cache_state:
        delta[name] = (cache_state[name] - trained_state[name]).cpu()

    avg_loss = total_loss / max(n_batches, 1)
    return delta, avg_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_at_model_size(global_model: FIARSEResNet,
                           model_size: float,
                           data_loader,
                           device: torch.device,
                           bn_calibration_loader=None) -> float:
    """
    Evaluate the global model at a specific model_size (parameter fraction).

    Creates a deep copy, generates mask at the given model_size, optionally
    calibrates BN, then evaluates.
    """
    eval_model = copy.deepcopy(global_model)
    eval_model.to(device)
    eval_model.generate_mask(model_size=model_size, topk=True, bern=False)

    if bn_calibration_loader is not None:
        calibrate_bn_running_stats(eval_model, bn_calibration_loader, device)

    eval_model.eval()
    correct, total = 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        preds = eval_model(x).argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    del eval_model
    torch.cuda.empty_cache()
    return 100.0 * correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    logging.info(f"Device: {device}")
    logging.info(f"FIARSE: unstructured magnitude pruning with TCB-GD, "
                 f"lr={args.lr}, lr_global={args.lr_global}")

    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               config=vars(args))

    # Client budgets (needed for γ-correlated data allocation)
    client_budgets_dict = get_client_budgets(
        num_clients=args.num_clients,
        max_mac=args.resource_max_mac,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    client_budgets_list = list(client_budgets_dict.values())
    client_model_sizes = [budget_to_model_size(b)
                          for b in client_budgets_list]

    logging.info(f"Model size range: [{min(client_model_sizes):.4f}, "
                 f"{max(client_model_sizes):.4f}]")

    num_classes = {'cifar100': 100, 'cinic10': 10, 'tinyimagenet': 200}[args.dataset]

    # Data (reuse feast data pipeline)
    logging.info(f"Loading {args.dataset.upper()} partitioned data...")
    _load_kwargs = dict(
        partition_method='hetero',
        partition_alpha=args.partition_alpha,
        client_number=args.num_clients,
        batch_size=args.batch_size,
        bn_calibration_split=0.1,
        client_budgets=client_budgets_dict,
        corr_gamma=args.corr_gamma,
        max_training_mac=args.max_training_mac,
    )
    if args.dataset == 'cifar100':
        (
            _train_num, _val_num, _test_num,
            _train_global, val_global, test_global,
            local_num_dict, train_data_local_dict, _test_data_local_dict,
            _class_num, bn_calibration_global, _bn_cal_num,
        ) = load_partition_data_cifar100(
            'CIFAR100', args.data_dir,
            validation_split=args.validation_split,
            augmentation=getattr(args, 'augmentation', 'basic'),
            **_load_kwargs,
        )
    elif args.dataset == 'cinic10':
        (
            _train_num, _val_num, _test_num,
            _train_global, val_global, test_global,
            local_num_dict, train_data_local_dict, _test_data_local_dict,
            _class_num, bn_calibration_global, _bn_cal_num,
        ) = load_partition_data_cinic10(
            'CINIC10', args.data_dir,
            augmentation=getattr(args, 'augmentation', 'basic'),
            **_load_kwargs,
        )
    else:  # tinyimagenet
        (
            _train_num, _val_num, _test_num,
            _train_global, val_global, test_global,
            local_num_dict, train_data_local_dict, _test_data_local_dict,
            _class_num, bn_calibration_global, _bn_cal_num,
        ) = load_partition_data_tinyimagenet(
            'TINYIMAGENET', args.data_dir,
            validation_split=args.validation_split,
            augmentation=getattr(args, 'augmentation', 'basic'),
            **_load_kwargs,
        )

    stem_stride  = getattr(args, 'stem_stride', 1)
    mix_mode     = getattr(args, 'mix_aug_mode', 'none')
    mixup_alpha  = getattr(args, 'mixup_alpha', 0.4)
    cutmix_alpha = getattr(args, 'cutmix_alpha', 1.0)
    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    criterion    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Global model
    global_model = build_global_model(num_classes=num_classes,
                                      stem_stride=stem_stride)
    global_model.to(device)
    federation = FIARSEFederation(global_model, lr_global=args.lr_global)

    # Checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_latest = os.path.join(args.checkpoint_dir, CKPT_LATEST)
    ckpt_best   = os.path.join(args.checkpoint_dir, CKPT_BEST)

    start_round = 0
    best_full_acc = 0.0
    if args.resume and os.path.exists(ckpt_latest):
        start_round, best_full_acc = load_checkpoint(ckpt_latest, global_model)
        start_round += 1
        logging.info(f"Resumed from round {start_round}, "
                     f"best_full_acc={best_full_acc:.2f}%")

    # Training rounds
    for round_idx in range(start_round, args.comm_round):
        if getattr(args, 'lr_cosine', False):
            cur_lr = args.lr * 0.5 * (1 + math.cos(math.pi * round_idx / max(args.comm_round, 1)))
        else:
            cur_lr = args.lr

        active_idxs = random.sample(range(args.num_clients),
                                    args.clients_per_round)

        delta_list = []
        sample_weights = []
        round_losses = []

        global_model.train()

        for client_idx in active_idxs:
            m_size = client_model_sizes[client_idx]

            # Distribute: deep copy + generate mask
            client_model = federation.distribute(m_size, device)

            # Local training
            delta, client_loss = train_one_client(
                client_model=client_model,
                data_loader=train_data_local_dict[client_idx],
                lr=cur_lr,
                local_epochs=args.local_epochs,
                device=device,
                criterion=criterion,
                mix_mode=mix_mode,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
            )

            del client_model
            torch.cuda.empty_cache()
            gc.collect()

            delta_list.append(delta)
            sample_weights.append(local_num_dict[client_idx])
            round_losses.append(client_loss)

        # Aggregate deltas
        federation.combine(delta_list, sample_weights)

        avg_train_loss = sum(round_losses) / len(round_losses)
        log_dict = {'round': round_idx, 'train_loss': avg_train_loss}

        # Periodic evaluation
        if (round_idx + 1) % args.eval_freq == 0 or round_idx == 0:
            logging.info(f"Round {round_idx+1}/{args.comm_round} "
                         f"(train_loss={avg_train_loss:.4f}) — evaluating...")
            for ms in EVAL_MODEL_SIZES:
                acc = evaluate_at_model_size(
                    global_model=global_model,
                    model_size=ms,
                    data_loader=val_global,
                    device=device,
                    bn_calibration_loader=bn_calibration_global,
                )
                log_dict[f'val_acc_ms{ms:.4f}'] = acc
                logging.info(
                    f"  model_size={ms:.4f} "
                    f"(~{ms * FULL_MODEL_MACS / 1e6:.0f}M MACs): {acc:.2f}%")

            full_acc = log_dict.get('val_acc_ms1.0000', 0.0)
            if full_acc > best_full_acc:
                best_full_acc = full_acc
                save_checkpoint(ckpt_best, round_idx, global_model,
                                best_full_acc)
                logging.info(
                    f"  *** New best full-model acc: {best_full_acc:.2f}% "
                    f"— saved {ckpt_best}")
            log_dict['best_full_acc'] = best_full_acc

        if (round_idx + 1) % args.save_freq == 0:
            save_checkpoint(ckpt_latest, round_idx, global_model,
                            best_full_acc)
            logging.info(f"  Checkpoint saved at round {round_idx+1}")

        wandb.log(log_dict)

    # Final test evaluation
    logging.info("=== Final test evaluation ===")
    final_results = {}
    for ms in EVAL_MODEL_SIZES:
        acc = evaluate_at_model_size(
            global_model=global_model,
            model_size=ms,
            data_loader=test_global,
            device=device,
            bn_calibration_loader=bn_calibration_global,
        )
        final_results[ms] = acc
        logging.info(
            f"  model_size={ms:.4f} "
            f"(~{ms * FULL_MODEL_MACS / 1e6:.0f}M MACs): {acc:.2f}%")

    wandb.log({f'final_ms{k:.4f}': v for k, v in final_results.items()})
    wandb.finish()
    return final_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='FIARSE on CIFAR-100 / CINIC-10 / TinyImageNet')
    # Dataset
    p.add_argument('--dataset',           type=str,   default='cifar100',
                   choices=['cifar100', 'cinic10', 'tinyimagenet'])
    # Federated
    p.add_argument('--num_clients',       type=int,   default=100)
    p.add_argument('--clients_per_round', type=int,   default=10)
    p.add_argument('--comm_round',        type=int,   default=2000)
    p.add_argument('--local_epochs',      type=int,   default=1)
    # Optimiser — following paper Table 3: no momentum, no weight_decay
    p.add_argument('--lr',                type=float, default=0.05,
                   help='Local learning rate (paper: ~0.05 for CIFAR)')
    p.add_argument('--lr_global',         type=float, default=1.0,
                   help='Server learning rate (paper default: 1.0)')
    p.add_argument('--batch_size',        type=int,   default=64)
    # Data
    p.add_argument('--data_dir',          type=str,   required=True)
    p.add_argument('--partition_alpha',   type=float, default=0.1)
    p.add_argument('--validation_split',  type=float, default=0.1)
    p.add_argument('--corr_gamma',        type=float, default=1.0)
    p.add_argument('--max_training_mac',  type=float, default=600_000_000)
    # Resource distribution
    p.add_argument('--resource_max_mac',  type=float, default=1_500_000_000)
    p.add_argument('--zipf_alpha',        type=float, default=1.2)
    p.add_argument('--seed',              type=int,   default=0)
    # Eval
    p.add_argument('--eval_freq',         type=int,   default=20)
    p.add_argument('--gpu',               type=int,   default=0)
    # Checkpointing
    p.add_argument('--checkpoint_dir',    type=str,
                   default='checkpoints/fiarse')
    p.add_argument('--save_freq',         type=int,   default=100)
    p.add_argument('--resume',            action='store_true')
    # WandB
    p.add_argument('--wandb_project',     type=str,
                   default='hetero_fednas_baselines')
    p.add_argument('--wandb_run_name',    type=str,
                   default='fiarse-gamma1')
    # Augmentation / regularisation
    p.add_argument('--stem_stride',       type=int,   default=1,
                   help='Stem conv stride (2 for TinyImageNet 64×64 → 32×32)')
    p.add_argument('--augmentation',      type=str,   default='basic',
                   choices=['basic', 'strong', 'mixaug'])
    p.add_argument('--label_smoothing',   type=float, default=0.0)
    p.add_argument('--lr_cosine',         action='store_true', default=False)
    p.add_argument('--mix_aug_mode',      type=str,   default='none',
                   choices=['none', 'mixup', 'cutmix', 'alternating'])
    p.add_argument('--mixup_alpha',       type=float, default=0.4)
    p.add_argument('--cutmix_alpha',      type=float, default=1.0)

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
