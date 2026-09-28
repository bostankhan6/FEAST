"""
ScaleFL ResNet for CIFAR-100.

Faithful implementation following:
  Ilhan et al., "ScaleFL: Resource-Adaptive Federated Learning with
  Heterogeneous Clients", CVPR 2023.
  Official repo: https://github.com/git-disl/scale-fl

Architecture: same ResNet backbone as HeteroFL (hidden=[64,128,256,512],
2 BasicBlocks per stage, CIFAR-adapted stem).  Early exit classifiers are
inserted after specific blocks as determined by the split-ratio search in
split_config.py.

2-D splitting:
  - Depth (s_d): exit placed after n_blocks[l] blocks (blocks counted from 0)
  - Width (s_w): top-left channel slice of every weight tensor

For the GLOBAL model (level=L=4): all 8 blocks + all 4 exit classifiers.
For a CLIENT model at level l:
  - Blocks 0 .. n_blocks[l]-1  (width s_w[l])
  - Exit classifiers 0 .. l-1  (input width s_w[l])
  - Forward pass returns a list [logits_exit_0, ..., logits_exit_{l-1}]
"""

import math
import torch
import torch.nn as nn
from typing import List

from .split_config import (
    get_default_configs, exit_channels,
    HIDDEN_SIZE, N_BLOCKS_PER_STAGE, TOTAL_BLOCKS,
)


# ---------------------------------------------------------------------------
# Block topology (fixed for our ResNet)
# ---------------------------------------------------------------------------
#
# Each tuple: (in_hidden_idx, out_hidden_idx, stride)
#   in_hidden_idx  → index into HIDDEN_SIZE for the block's input channels
#   out_hidden_idx → index into HIDDEN_SIZE for the block's output channels
#   stride         → spatial downsampling factor (1 or 2)
#
BLOCK_SPECS = [
    (0, 0, 1),   # stage 0, block 0
    (0, 0, 1),   # stage 0, block 1
    (0, 1, 2),   # stage 1, block 0  (64→128, stride=2)
    (1, 1, 1),   # stage 1, block 1
    (1, 2, 2),   # stage 2, block 0  (128→256, stride=2)
    (2, 2, 1),   # stage 2, block 1
    (2, 3, 2),   # stage 3, block 0  (256→512, stride=2)
    (3, 3, 1),   # stage 3, block 1
]


# ---------------------------------------------------------------------------
# BasicBlock (no Scaler — ScaleFL does not use HeteroFL's Scaler)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 track_running_stats: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=None,
                                  track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=None,
                                  track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=None,
                               track_running_stats=track_running_stats),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# ---------------------------------------------------------------------------
# Exit classifier
# ---------------------------------------------------------------------------

def _make_exit_classifier(in_ch: int, num_classes: int) -> nn.Sequential:
    """GlobalAvgPool → Flatten → Linear exit head."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_ch, num_classes),
    )


# ---------------------------------------------------------------------------
# ScaleFL ResNet
# ---------------------------------------------------------------------------

class ScaleFLResNet(nn.Module):
    """
    ResNet with early exits for ScaleFL.

    Args:
        n_blocks:        Number of backbone blocks to include (1–8).
        s_w:             Width scaling ratio.  All hidden channels scaled by
                         ceil(s_w * base_channels).
        exit_positions:  Sorted list of block indices (0-based) where an exit
                         classifier is placed.  Must satisfy
                         exit_positions[-1] == n_blocks - 1.
        num_classes:     Output classes (100 for CIFAR-100).

    forward(x) returns a list of logit tensors, one per exit in order.
    """

    def __init__(self, n_blocks: int, s_w: float,
                 exit_positions: List[int], num_classes: int = 100,
                 stem_stride: int = 1):
        super().__init__()
        assert exit_positions[-1] == n_blocks - 1, (
            f"Last exit position ({exit_positions[-1]}) must equal "
            f"n_blocks-1 ({n_blocks-1})"
        )

        self.n_blocks       = n_blocks
        self.s_w            = s_w
        self.exit_positions = exit_positions
        self.num_classes    = num_classes

        hs = [max(1, math.ceil(s_w * c)) for c in HIDDEN_SIZE]

        # Stem: stride=1 for 32×32 (CIFAR/CINIC), stride=2 for 64×64 (TinyImageNet)
        self.stem = nn.Sequential(
            nn.Conv2d(3, hs[0], 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(hs[0], momentum=None, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        # Backbone blocks
        self.blocks = nn.ModuleList()
        for blk_idx in range(n_blocks):
            in_h_idx, out_h_idx, stride = BLOCK_SPECS[blk_idx]
            in_ch  = hs[in_h_idx]
            out_ch = hs[out_h_idx]
            # Correct in_ch for stem output at block 0
            if blk_idx == 0:
                in_ch = hs[0]
            self.blocks.append(
                BasicBlock(in_ch, out_ch, stride=stride)
            )

        # Exit classifiers (one per exit position)
        self.exit_classifiers = nn.ModuleList()
        for pos in exit_positions:
            _, out_h_idx, _ = BLOCK_SPECS[pos]
            ch = hs[out_h_idx]
            self.exit_classifiers.append(
                _make_exit_classifier(ch, num_classes)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of logit tensors [exit_0, exit_1, ..., exit_{L-1}]."""
        out = self.stem(x)
        outputs   = []
        exit_idx  = 0
        for blk_idx, block in enumerate(self.blocks):
            out = block(out)
            if (exit_idx < len(self.exit_positions)
                    and blk_idx == self.exit_positions[exit_idx]):
                outputs.append(self.exit_classifiers[exit_idx](out))
                exit_idx += 1
        return outputs


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def _build_exit_positions(target_level: int,
                          all_n_blocks: List[int]) -> List[int]:
    """Exit positions for a client at target_level (1-based).

    Exits fire at block indices all_n_blocks[l]-1 for l = 0..target_level-1.
    """
    return [all_n_blocks[l] - 1 for l in range(target_level)]


def build_global_model(num_classes: int = 100, stem_stride: int = 1) -> ScaleFLResNet:
    """Full-width global model (level=4, s_w=1.0, all exits)."""
    configs, _ = get_default_configs()
    all_n_blocks = [c['n_blocks'] for c in configs]
    exit_pos     = _build_exit_positions(len(configs), all_n_blocks)
    return ScaleFLResNet(
        n_blocks=TOTAL_BLOCKS, s_w=1.0,
        exit_positions=exit_pos, num_classes=num_classes,
        stem_stride=stem_stride,
    )


def build_client_model(level: int, num_classes: int = 100,
                       stem_stride: int = 1) -> ScaleFLResNet:
    """Client model for complexity level `level` (1-based)."""
    configs, _ = get_default_configs()
    cfg          = configs[level - 1]
    all_n_blocks = [c['n_blocks'] for c in configs]
    exit_pos     = _build_exit_positions(level, all_n_blocks)
    return ScaleFLResNet(
        n_blocks=cfg['n_blocks'], s_w=cfg['s_w'],
        exit_positions=exit_pos, num_classes=num_classes,
        stem_stride=stem_stride,
    )
