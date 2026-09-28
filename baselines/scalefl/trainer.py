"""
ScaleFL Training Loop for CIFAR-100.

Faithful to ScaleFL paper (Ilhan et al., CVPR 2023):
  - 2D model splitting (depth + width) with early-exit classifiers
  - Self-distillation loss (Eq. 5): final exit = teacher, earlier = students
    L = (1/(l*(l+1))) * sum_{i=1}^{l} i * (beta * KL(exit_i, exit_l) + CE(exit_i, y))
  - Temperature tau=3, beta=0.1 (image classification defaults from paper)
  - Same aggregation: top-left slice averaging (Eq. 2, identical to HeteroFL)
  - BN: track_running_stats=False

Data pipeline reuses feast load_partition_data_cifar100 for identical
data splits across all methods (same Dirichlet + gamma-correlated allocation).
"""

import sys
import os
import logging
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

CKPT_LATEST = 'checkpoint_latest.pt'
CKPT_BEST   = 'best_full.pt'

import math
import numpy as _np

from feast.data.cifar100.data_loader import load_partition_data_cifar100
from feast.data.cinic10.data_loader import load_partition_data_cinic10
from feast.data.tinyimagenet.data_loader import load_partition_data_tinyimagenet

from .model import ScaleFLResNet, build_global_model, build_client_model
from .federation import (ScaleFLFederation, budget_to_level,
                          LEVEL_MACS, S_W_PER_LEVEL, N_LEVELS)
from .split_config import get_default_configs
from baselines.utils import get_client_budgets
from baselines.bn_calibration import calibrate_bn_running_stats


# ---------------------------------------------------------------------------
# Self-distillation loss  (Eq. 5)
# ---------------------------------------------------------------------------

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


def scalefl_loss(outputs: list, labels: torch.Tensor,
                 beta: float = 0.1, tau: float = 3.0,
                 label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Self-distillation loss for ScaleFL (Eq. 5).

    L = (1/(l*(l+1))) * sum_{i=1}^{l} i * (beta * L_KL(exit_i, exit_l) + L_CE(exit_i, y))

    Where exit_l (the deepest exit) is the teacher and earlier exits are
    students.  KL divergence includes temperature scaling and the tau^2
    gradient-magnitude correction standard in knowledge distillation.

    Args:
        outputs: list of logit tensors [exit_0, ..., exit_{l-1}]
        labels:  ground-truth LongTensor (B,)
        beta:    KL weight (paper default = 0.1 for image tasks)
        tau:     distillation temperature (paper default = 3)
        label_smoothing: CE label smoothing (0.0 = paper default)
    """
    l = len(outputs)
    teacher_logits = outputs[-1]
    criterion_ce   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total = torch.tensor(0.0, device=outputs[0].device)

    for i, logits in enumerate(outputs):
        weight = (i + 1) / (l * (l + 1))

        ce = criterion_ce(logits, labels)

        if i < l - 1:
            log_p_s = F.log_softmax(logits          / tau, dim=1)
            p_t     = F.softmax(teacher_logits.detach() / tau, dim=1)
            kl = (tau ** 2) * F.kl_div(log_p_s, p_t, reduction='batchmean')
        else:
            kl = torch.tensor(0.0, device=outputs[0].device)

        total = total + weight * (beta * kl + ce)

    return total


# ---------------------------------------------------------------------------
# Client budget assignment
# ---------------------------------------------------------------------------

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




# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_one_client(client_model: ScaleFLResNet,
                     local_params: dict,
                     data_loader,
                     lr: float,
                     local_epochs: int,
                     device: torch.device,
                     beta: float = 0.1,
                     tau:  float = 3.0,
                     wd: float = 4e-5,
                     label_smoothing: float = 0.0,
                     mix_mode: str = 'none',
                     mixup_alpha: float = 0.4,
                     cutmix_alpha: float = 1.0):
    sd = client_model.state_dict()
    for name, param in local_params.items():
        if name in sd:
            sd[name].copy_(param)
    client_model.load_state_dict(sd)

    client_model.to(device)
    client_model.train()

    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=wd)

    total_loss, n_batches = 0.0, 0
    for _ in range(local_epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            x_mix, y_a, y_b, lam = _mix_batch(x, y, mix_mode, mixup_alpha,
                                               cutmix_alpha, device)
            optimizer.zero_grad()
            outputs = client_model(x_mix)
            # Apply mixup linearly over the ScaleFL composite loss
            loss = (lam * scalefl_loss(outputs, y_a, beta=beta, tau=tau,
                                       label_smoothing=label_smoothing) +
                    (1 - lam) * scalefl_loss(outputs, y_b, beta=beta, tau=tau,
                                             label_smoothing=label_smoothing))
            loss.backward()
            nn.utils.clip_grad_norm_(client_model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    updated = {name: client_model.state_dict()[name].cpu().clone()
               for name in local_params}
    avg_loss = total_loss / max(n_batches, 1)
    return updated, avg_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_level(level: int, global_model: ScaleFLResNet,
                   federation: ScaleFLFederation,
                   data_loader, device: torch.device,
                   bn_calibration_loader=None,
                   num_classes: int = 100,
                   stem_stride: int = 1) -> float:
    local_params, _ = federation.distribute(level)
    client_model    = build_client_model(level, num_classes=num_classes,
                                         stem_stride=stem_stride)

    sd = client_model.state_dict()
    for name in local_params:
        if name in sd:
            sd[name].copy_(local_params[name])
    client_model.load_state_dict(sd)
    client_model.to(device)

    if bn_calibration_loader is not None:
        calibrate_bn_running_stats(client_model, bn_calibration_loader, device)

    client_model.eval()
    correct, total = 0, 0
    for x, y in data_loader:
        x, y   = x.to(device), y.to(device)
        outputs = client_model(x)
        preds   = outputs[-1].argmax(dim=1)   # deepest exit
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

    # Print split configs for reproducibility
    configs, full_macs = get_default_configs()
    logging.info(f"Full model MACs: {full_macs/1e6:.1f}M")
    for c in configs:
        logging.info(f"  Level {c['level']}: n_blocks={c['n_blocks']}, "
                     f"s_d={c['s_d']:.3f}, s_w={c['s_w']:.2f}, "
                     f"MACs={c['macs']/1e6:.1f}M (r={c['r_actual']:.3f})")

    # WandB
    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               config=vars(args))

    # Client budgets (needed for gamma-correlated data allocation)
    client_budgets_dict = get_client_budgets(
        num_clients=args.num_clients,
        max_mac=args.resource_max_mac,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    client_budgets_list = list(client_budgets_dict.values())

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

    # Level assignment
    client_levels = [budget_to_level(b) for b in client_budgets_list]
    level_counts  = {l: client_levels.count(l) for l in range(1, N_LEVELS + 1)}
    logging.info(f"Level distribution: {level_counts}")

    stem_stride     = getattr(args, 'stem_stride', 1)
    mix_mode        = getattr(args, 'mix_aug_mode', 'none')
    mixup_alpha     = getattr(args, 'mixup_alpha', 0.4)
    cutmix_alpha    = getattr(args, 'cutmix_alpha', 1.0)
    wd              = getattr(args, 'wd', 4e-5)
    label_smoothing = getattr(args, 'label_smoothing', 0.0)

    # Global model + federation
    global_model = build_global_model(num_classes=num_classes,
                                      stem_stride=stem_stride)
    global_model.to(device)
    federation   = ScaleFLFederation(global_model)

    # --- Checkpoint dir ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_latest = os.path.join(args.checkpoint_dir, CKPT_LATEST)
    ckpt_best   = os.path.join(args.checkpoint_dir, CKPT_BEST)

    start_round = 0
    best_full_acc = 0.0
    if args.resume and os.path.exists(ckpt_latest):
        start_round, best_full_acc = load_checkpoint(ckpt_latest, global_model)
        start_round += 1
        logging.info(f"Resumed from checkpoint at round {start_round}, "
                     f"best_full_acc={best_full_acc:.2f}%")

    # Training rounds
    for round_idx in range(start_round, args.comm_round):
        if getattr(args, 'lr_cosine', False):
            cur_lr = args.lr * 0.5 * (1 + math.cos(math.pi * round_idx / max(args.comm_round, 1)))
        else:
            cur_lr = args.lr

        active_idxs = random.sample(range(args.num_clients),
                                    args.clients_per_round)

        local_params_list = []
        param_idx_list    = []
        sample_weights    = []
        round_losses      = []

        for client_idx in active_idxs:
            level        = client_levels[client_idx]
            client_model = build_client_model(level, num_classes=num_classes,
                                              stem_stride=stem_stride)

            local_params, param_idx = federation.distribute(level)

            updated, client_loss = train_one_client(
                client_model    = client_model,
                local_params    = local_params,
                data_loader     = train_data_local_dict[client_idx],
                lr              = cur_lr,
                local_epochs    = args.local_epochs,
                device          = device,
                beta            = args.beta,
                tau             = args.tau,
                wd              = wd,
                label_smoothing = label_smoothing,
                mix_mode        = mix_mode,
                mixup_alpha     = mixup_alpha,
                cutmix_alpha    = cutmix_alpha,
            )

            local_params_list.append(updated)
            param_idx_list.append(param_idx)
            sample_weights.append(local_num_dict[client_idx])
            round_losses.append(client_loss)

        federation.combine(local_params_list, param_idx_list, sample_weights)

        avg_train_loss = sum(round_losses) / len(round_losses)
        log_dict = {'round': round_idx, 'train_loss': avg_train_loss}

        if (round_idx + 1) % args.eval_freq == 0 or round_idx == 0:
            logging.info(f"Round {round_idx+1}/{args.comm_round} "
                         f"(train_loss={avg_train_loss:.4f}) — evaluating...")
            for level in range(1, N_LEVELS + 1):
                if level_counts.get(level, 0) == 0:
                    continue   # no client assigned — skip untrained level
                acc = evaluate_level(
                    level=level,
                    global_model=global_model,
                    federation=federation,
                    data_loader=val_global,
                    device=device,
                    bn_calibration_loader=bn_calibration_global,
                    num_classes=num_classes,
                    stem_stride=stem_stride,
                )
                log_dict[f'val_acc_level_{level}'] = acc
                logging.info(f"  Level {level} "
                             f"(~{LEVEL_MACS[level-1]/1e6:.0f}M MACs, "
                             f"s_w={S_W_PER_LEVEL[level-1]:.2f}): {acc:.2f}%")

            # Track best full-model (level 4) accuracy
            full_acc = log_dict.get(f'val_acc_level_{N_LEVELS}', 0.0)
            if full_acc > best_full_acc:
                best_full_acc = full_acc
                save_checkpoint(ckpt_best, round_idx, global_model, best_full_acc)
                logging.info(f"  *** New best full-model acc: {best_full_acc:.2f}% "
                             f"— saved {ckpt_best}")
            log_dict['best_full_acc'] = best_full_acc

        if (round_idx + 1) % args.save_freq == 0:
            save_checkpoint(ckpt_latest, round_idx, global_model, best_full_acc)
            logging.info(f"  Checkpoint saved at round {round_idx+1}")

        wandb.log(log_dict)

    # Final test evaluation
    logging.info("=== Final test evaluation ===")
    final_results = {}
    for level in range(1, N_LEVELS + 1):
        if level_counts.get(level, 0) == 0:
            continue
        acc = evaluate_level(
            level=level,
            global_model=global_model,
            federation=federation,
            data_loader=test_global,
            device=device,
            bn_calibration_loader=bn_calibration_global,
            num_classes=num_classes,
            stem_stride=stem_stride,
        )
        final_results[level] = acc
        logging.info(f"  Level {level} (~{LEVEL_MACS[level-1]/1e6:.0f}M MACs): "
                     f"{acc:.2f}%")

    wandb.log({f'final_level_{k}': v for k, v in final_results.items()})
    wandb.finish()
    return final_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='ScaleFL on CIFAR-100 / CINIC-10 / TinyImageNet')
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
    # ScaleFL hyperparams (paper defaults)
    p.add_argument('--beta',              type=float, default=0.1,
                   help='KL weight in self-distillation loss')
    p.add_argument('--tau',               type=float, default=3.0,
                   help='Temperature for self-distillation')
    # Eval
    p.add_argument('--eval_freq',         type=int,   default=20)
    p.add_argument('--gpu',               type=int,   default=0)
    # Checkpointing
    p.add_argument('--checkpoint_dir',    type=str,   default='checkpoints/scalefl')
    p.add_argument('--save_freq',         type=int,   default=100,
                   help='Save latest checkpoint every N rounds')
    p.add_argument('--resume',            action='store_true',
                   help='Resume from latest checkpoint if it exists')
    # WandB
    p.add_argument('--wandb_project',     type=str,   default='hetero_fednas_baselines')
    p.add_argument('--wandb_run_name',    type=str,   default='scalefl-gamma1')
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
