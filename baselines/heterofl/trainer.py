"""
HeteroFL Training Loop for CIFAR-100.

Plugs into feast data pipeline (same Dirichlet + γ-correlated allocation)
so comparisons with our OFA method are on identical data splits.

Faithful to HeteroFL paper:
  - Clients assigned to tiers by MAC budget (MACs-based, not quantile)
  - Local training: SGD + CrossEntropy, standard backprop
  - Aggregation: HeteroFLFederation.combine() (top-left slice averaging)
  - BN: track_running_stats=False (batch stats only, no server-side calibration)
  - Evaluation: server runs BN calibration pass then evaluates all 5 tier models
"""

import sys
import os
import copy
import logging
import argparse
import json

import torch
import torch.nn as nn
import wandb

# feast data pipeline (same splits for all methods)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

CKPT_LATEST = 'checkpoint_latest.pt'
CKPT_BEST   = 'best_full.pt'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import math
import numpy as _np

from feast.data.cifar100.data_loader import load_partition_data_cifar100
from feast.data.cinic10.data_loader import load_partition_data_cinic10
from feast.data.tinyimagenet.data_loader import load_partition_data_tinyimagenet

from .model import build_global_model, build_heterofl_model, HeteroFLResNet
from .federation import HeteroFLFederation, budget_to_tier, TIER_RATES, TIER_MACS
from baselines.utils import get_client_budgets
from baselines.bn_calibration import calibrate_bn_running_stats


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


def save_checkpoint(path: str, round_idx: int, global_model, best_acc: float):
    torch.save({
        'round': round_idx,
        'model_state_dict': global_model.state_dict(),
        'best_full_acc': best_acc,
    }, path)


def load_checkpoint(path: str, global_model):
    ckpt = torch.load(path, map_location='cpu')
    global_model.load_state_dict(ckpt['model_state_dict'])
    return ckpt['round'], ckpt['best_full_acc']


def train_one_client(model: HeteroFLResNet,
                     local_params: dict,
                     data_loader,
                     lr: float,
                     local_epochs: int,
                     device: torch.device,
                     criterion: nn.Module,
                     wd: float = 4e-5,
                     mix_mode: str = 'none',
                     mixup_alpha: float = 0.4,
                     cutmix_alpha: float = 1.0):
    client_sd = model.state_dict()
    for name, param in local_params.items():
        if name in client_sd:
            client_sd[name].copy_(param)
    model.load_state_dict(client_sd)

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)

    total_loss, n_batches = 0.0, 0
    for _ in range(local_epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            x_mix, y_a, y_b, lam = _mix_batch(x, y, mix_mode, mixup_alpha,
                                               cutmix_alpha, device)
            optimizer.zero_grad()
            logits = model(x_mix)
            loss   = _mixed_ce(criterion, logits, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    updated_params = {name: model.state_dict()[name].cpu().clone()
                      for name in local_params}
    avg_loss = total_loss / max(n_batches, 1)
    return updated_params, avg_loss


@torch.no_grad()
def evaluate(global_model: HeteroFLResNet, data_loader,
             device: torch.device) -> float:
    """Evaluate global model (rate=1.0) on data_loader."""
    global_model.eval()
    global_model.to(device)
    correct, total = 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        preds = global_model(x).argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_tier(tier_name: str, global_model: HeteroFLResNet,
                  federation: HeteroFLFederation,
                  data_loader, device: torch.device,
                  bn_calibration_loader=None,
                  num_classes: int = 100,
                  stem_stride: int = 1) -> float:
    local_params, param_idx = federation.distribute(tier_name)
    rate = TIER_RATES[tier_name]
    tier_model = build_heterofl_model(rate, num_classes=num_classes,
                                      stem_stride=stem_stride)

    # Load distributed params into tier model
    sd = tier_model.state_dict()
    for name in local_params:
        if name in sd:
            sd[name].copy_(local_params[name])
    tier_model.load_state_dict(sd)
    tier_model.to(device)

    # BN calibration: proper running-stats calibration (not a no-op)
    if bn_calibration_loader is not None:
        calibrate_bn_running_stats(tier_model, bn_calibration_loader, device)

    tier_model.eval()
    correct, total = 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        preds = tier_model(x).argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total   += y.size(0)
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

    # --- WandB ---
    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               config=vars(args))

    num_classes = {'cifar100': 100, 'cinic10': 10, 'tinyimagenet': 200}[args.dataset]

    # --- Client budgets first (needed for γ-correlated data allocation) ---
    client_budgets_dict = get_client_budgets(
        num_clients=args.num_clients,
        max_mac=args.resource_max_mac,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    client_budgets_list = list(client_budgets_dict.values())

    # --- Data (reuse feast data pipeline) ---
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
            local_num_dict, train_data_local_dict, test_data_local_dict,
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
            local_num_dict, train_data_local_dict, test_data_local_dict,
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
            local_num_dict, train_data_local_dict, test_data_local_dict,
            _class_num, bn_calibration_global, _bn_cal_num,
        ) = load_partition_data_tinyimagenet(
            'TINYIMAGENET', args.data_dir,
            validation_split=args.validation_split,
            augmentation=getattr(args, 'augmentation', 'basic'),
            **_load_kwargs,
        )

    # --- Tier assignment ---
    client_tiers = [budget_to_tier(b) for b in client_budgets_list]
    tier_counts  = {t: client_tiers.count(t) for t in 'abcde'}
    logging.info(f"Tier distribution: {tier_counts}")

    stem_stride = getattr(args, 'stem_stride', 1)

    # --- Global model + federation ---
    global_model = build_global_model(num_classes=num_classes, use_scaler=True,
                                      stem_stride=stem_stride)
    global_model.to(device)
    federation   = HeteroFLFederation(global_model)

    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # --- Checkpoint dir ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_latest = os.path.join(args.checkpoint_dir, CKPT_LATEST)
    ckpt_best   = os.path.join(args.checkpoint_dir, CKPT_BEST)

    start_round = 0
    best_full_acc = 0.0
    if args.resume and os.path.exists(ckpt_latest):
        start_round, best_full_acc = load_checkpoint(ckpt_latest, global_model)
        start_round += 1   # resume from next round
        logging.info(f"Resumed from checkpoint at round {start_round}, "
                     f"best_full_acc={best_full_acc:.2f}%")

    mix_mode     = getattr(args, 'mix_aug_mode', 'none')
    mixup_alpha  = getattr(args, 'mixup_alpha', 0.4)
    cutmix_alpha = getattr(args, 'cutmix_alpha', 1.0)
    wd           = getattr(args, 'wd', 4e-5)

    # --- Training rounds ---
    for round_idx in range(start_round, args.comm_round):
        if getattr(args, 'lr_cosine', False):
            cur_lr = args.lr * 0.5 * (1 + math.cos(math.pi * round_idx / max(args.comm_round, 1)))
        else:
            cur_lr = args.lr

        # Sample active clients
        import random
        active_idxs = random.sample(range(args.num_clients),
                                    args.clients_per_round)

        local_params_list = []
        param_idx_list    = []
        sample_weights    = []
        round_losses      = []

        for client_idx in active_idxs:
            tier_name  = client_tiers[client_idx]
            rate       = TIER_RATES[tier_name]

            # Build client model at this tier's rate
            client_model = build_heterofl_model(rate, num_classes=num_classes,
                                                use_scaler=True,
                                                stem_stride=stem_stride)

            # Distribute global params to client
            local_params, param_idx = federation.distribute(tier_name)

            # Local training
            updated_params, client_loss = train_one_client(
                model        = client_model,
                local_params = local_params,
                data_loader  = train_data_local_dict[client_idx],
                lr           = cur_lr,
                local_epochs = args.local_epochs,
                device       = device,
                criterion    = criterion,
                wd           = wd,
                mix_mode     = mix_mode,
                mixup_alpha  = mixup_alpha,
                cutmix_alpha = cutmix_alpha,
            )

            local_params_list.append(updated_params)
            param_idx_list.append(param_idx)
            sample_weights.append(local_num_dict[client_idx])
            round_losses.append(client_loss)

        # Aggregate
        federation.combine(local_params_list, param_idx_list, sample_weights)

        # --- Logging ---
        avg_train_loss = sum(round_losses) / len(round_losses)
        log_dict = {'round': round_idx, 'train_loss': avg_train_loss}

        # Periodic evaluation
        if (round_idx + 1) % args.eval_freq == 0 or round_idx == 0:
            logging.info(f"Round {round_idx+1}/{args.comm_round} "
                         f"(train_loss={avg_train_loss:.4f}) — evaluating...")
            for tier_name in TIER_RATES:
                if tier_counts.get(tier_name, 0) == 0:
                    continue   # no client assigned — skip untrained tier
                acc = evaluate_tier(
                    tier_name=tier_name,
                    global_model=global_model,
                    federation=federation,
                    data_loader=val_global,
                    device=device,
                    bn_calibration_loader=bn_calibration_global,
                    num_classes=num_classes,
                    stem_stride=stem_stride,
                )
                log_dict[f'val_acc_tier_{tier_name}'] = acc
                logging.info(f"  Tier {tier_name} "
                             f"(rate={TIER_RATES[tier_name]:.4f}, "
                             f"~{TIER_MACS[tier_name]/1e6:.0f}M MACs): "
                             f"{acc:.2f}%")

            # Track best full-model accuracy and save best checkpoint
            full_acc = log_dict.get('val_acc_tier_a', 0.0)
            if full_acc > best_full_acc:
                best_full_acc = full_acc
                save_checkpoint(ckpt_best, round_idx, global_model, best_full_acc)
                logging.info(f"  *** New best full-model acc: {best_full_acc:.2f}% "
                             f"— saved {ckpt_best}")
            log_dict['best_full_acc'] = best_full_acc

        # Periodic checkpoint (every save_freq rounds)
        if (round_idx + 1) % args.save_freq == 0:
            save_checkpoint(ckpt_latest, round_idx, global_model, best_full_acc)
            logging.info(f"  Checkpoint saved at round {round_idx+1}")

        wandb.log(log_dict)

    # --- Final evaluation on test set ---
    logging.info("=== Final test evaluation ===")
    final_results = {}
    for tier_name in TIER_RATES:
        if tier_counts.get(tier_name, 0) == 0:
            continue
        acc = evaluate_tier(
            tier_name=tier_name,
            global_model=global_model,
            federation=federation,
            data_loader=test_global,
            device=device,
            bn_calibration_loader=bn_calibration_global,
            num_classes=num_classes,
            stem_stride=stem_stride,
        )
        final_results[tier_name] = acc
        logging.info(f"  Tier {tier_name} "
                     f"(~{TIER_MACS[tier_name]/1e6:.0f}M MACs): {acc:.2f}%")

    wandb.log({'final_' + k: v for k, v in final_results.items()})
    wandb.finish()
    return final_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='HeteroFL on CIFAR-100 / CINIC-10 / TinyImageNet')
    # Dataset
    p.add_argument('--dataset',           type=str,   default='cifar100',
                   choices=['cifar100', 'cinic10', 'tinyimagenet'])
    # Federated
    p.add_argument('--num_clients',       type=int,   default=100)
    p.add_argument('--clients_per_round', type=int,   default=10)
    p.add_argument('--comm_round',        type=int,   default=2000)
    p.add_argument('--local_epochs',      type=int,   default=1)
    # Optimiser
    p.add_argument('--lr',                type=float, default=0.025)
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
    p.add_argument('--checkpoint_dir',    type=str,   default='checkpoints/heterofl')
    p.add_argument('--save_freq',         type=int,   default=100,
                   help='Save latest checkpoint every N rounds')
    p.add_argument('--resume',            action='store_true',
                   help='Resume from latest checkpoint if it exists')
    # WandB
    p.add_argument('--wandb_project',     type=str,   default='hetero_fednas_baselines')
    p.add_argument('--wandb_run_name',    type=str,   default='heterofl-gamma1')
    # Augmentation / regularisation
    p.add_argument('--stem_stride',       type=int,   default=1,
                   help='Stem conv stride (2 for TinyImageNet 64×64 → 32×32)')
    p.add_argument('--augmentation',      type=str,   default='basic',
                   choices=['basic', 'strong', 'mixaug'])
    p.add_argument('--label_smoothing',   type=float, default=0.0)
    p.add_argument('--lr_cosine',         action='store_true', default=False)
    p.add_argument('--wd',                type=float, default=4e-5)
    p.add_argument('--mix_aug_mode',      type=str,   default='none',
                   choices=['none', 'mixup', 'cutmix', 'alternating'])
    p.add_argument('--mixup_alpha',       type=float, default=0.4)
    p.add_argument('--cutmix_alpha',      type=float, default=1.0)

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
