"""
ScaleFL 2D split configuration.

Finds the most balanced (depth, width) split pair for each complexity level
following Section 3.1.1 of Ilhan et al., CVPR 2023.

"Most balanced" means minimising |s_d - s_w|, i.e., scaling along both
dimensions as uniformly as possible.

Our ResNet: hidden_size=[64,128,256,512], 4 stages x 2 BasicBlocks each
(8 blocks total), CIFAR-100 32x32.  Exits are evaluated at every block
boundary; the search finds the best depth/width pair for each target ratio.
"""

import math
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Architecture constants — must match model.py
# ---------------------------------------------------------------------------

HIDDEN_SIZE        = [64, 128, 256, 512]
N_BLOCKS_PER_STAGE = 2
N_STAGES           = 4
TOTAL_BLOCKS       = N_STAGES * N_BLOCKS_PER_STAGE   # 8
INPUT_SIZE         = 32
NUM_CLASSES        = 100


# ---------------------------------------------------------------------------
# MAC counter
# ---------------------------------------------------------------------------

def compute_macs(n_blocks: int, s_w: float,
                 hidden_size: List[int] = HIDDEN_SIZE,
                 n_per_stage: int       = N_BLOCKS_PER_STAGE,
                 input_size: int        = INPUT_SIZE) -> int:
    """
    Compute MACs for the first n_blocks BasicBlocks with width ratio s_w.

    Includes:
    - CIFAR stem Conv2d(3, ceil(s_w*H[0]), 3x3, stride=1)
    - BasicBlocks with optional 2x downsampling (stride=2 at first block of
      each stage except stage 0)
    - Shortcut 1x1 conv when channels change or stride > 1

    Does NOT include exit-classifier MACs (minor, dominated by backbone).
    """
    total = 0

    # Stem
    stem_out = max(1, math.ceil(s_w * hidden_size[0]))
    total += 3 * stem_out * 9 * input_size * input_size

    in_ch   = stem_out
    spatial = input_size
    counted = 0

    for stage_idx, base_c in enumerate(hidden_size):
        out_ch       = max(1, math.ceil(s_w * base_c))
        stage_stride = 1 if stage_idx == 0 else 2

        for b in range(n_per_stage):
            counted += 1
            if counted > n_blocks:
                return total

            blk_stride = stage_stride if b == 0 else 1
            if blk_stride > 1:
                spatial = spatial // blk_stride

            # conv1 (in_ch → out_ch, 3x3)
            total += in_ch * out_ch * 9 * spatial * spatial
            # conv2 (out_ch → out_ch, 3x3)
            total += out_ch * out_ch * 9 * spatial * spatial
            # shortcut 1x1 (when channels change or stride > 1)
            if in_ch != out_ch or blk_stride > 1:
                total += in_ch * out_ch * 1 * spatial * spatial

            in_ch = out_ch

    return total


# ---------------------------------------------------------------------------
# Split-ratio search
# ---------------------------------------------------------------------------

def find_split_configs(
    target_ratios: List[float] = (0.061, 0.13, 0.25, 1.0),
    tolerance:     float        = 0.15,
    s_w_steps:     int          = 20,
) -> Tuple[List[Dict], int]:
    """
    For each target cost reduction ratio r_l, find the most balanced (s_d, s_w)
    pair such that  |actual_macs / target_macs - 1| <= tolerance.

    "Most balanced" = minimise |s_d - s_w|. With the defaults above this search
    reproduces the four (n_blocks, s_w) levels Supplementary Sec. E.1 states for
    ScaleFL: (3, 0.40), (4, 0.50), (5, 0.60), (8, 1.00).

    Returns:
        configs  : list of dicts (one per level), sorted by level ascending
        full_macs: MACs of the full model (n_blocks=TOTAL_BLOCKS, s_w=1.0)
    """
    full_macs  = compute_macs(TOTAL_BLOCKS, 1.0)
    s_w_cands  = [max(0.05, i / s_w_steps) for i in range(1, s_w_steps + 1)]

    configs = []
    for level, r_l in enumerate(target_ratios, start=1):
        if r_l >= 1.0:
            configs.append({
                'level': level, 'n_blocks': TOTAL_BLOCKS,
                's_d': 1.0, 's_w': 1.0,
                'macs': full_macs, 'r_actual': 1.0,
            })
            continue

        target_macs = r_l * full_macs

        best            = None
        best_imbalance  = float('inf')

        for nb in range(1, TOTAL_BLOCKS + 1):
            s_d = nb / TOTAL_BLOCKS
            for s_w in s_w_cands:
                macs    = compute_macs(nb, s_w)
                rel_err = abs(macs / target_macs - 1.0)
                if rel_err <= tolerance:
                    imbalance = abs(s_d - s_w)
                    if imbalance < best_imbalance:
                        best_imbalance = imbalance
                        best = {
                            'level': level, 'n_blocks': nb,
                            's_d': s_d, 's_w': s_w,
                            'macs': macs, 'r_actual': macs / full_macs,
                        }

        if best is None:
            # Tolerance not met — fall back to closest absolute match
            best = min(
                ({'level': level, 'n_blocks': nb, 's_d': nb / TOTAL_BLOCKS,
                  's_w': s_w, 'macs': compute_macs(nb, s_w),
                  'r_actual': compute_macs(nb, s_w) / full_macs}
                 for nb in range(1, TOTAL_BLOCKS + 1)
                 for s_w in s_w_cands),
                key=lambda x: abs(x['r_actual'] - r_l),
            )

        configs.append(best)

    return configs, full_macs


# ---------------------------------------------------------------------------
# Channel count at the exit position for a given split config
# ---------------------------------------------------------------------------

def exit_channels(n_blocks: int, s_w: float,
                  hidden_size: List[int] = HIDDEN_SIZE,
                  n_per_stage: int       = N_BLOCKS_PER_STAGE) -> int:
    """
    Return the output channel count at the n_blocks-th block with width s_w.
    This is the input size for the exit classifier at that position.
    """
    block = 0
    for stage_idx, base_c in enumerate(hidden_size):
        for _ in range(n_per_stage):
            block += 1
            if block == n_blocks:
                return max(1, math.ceil(s_w * base_c))
    return max(1, math.ceil(s_w * hidden_size[-1]))


# ---------------------------------------------------------------------------
# Cached default configs (computed once)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS: List[Dict] = []
_FULL_MACS:       int        = 0


def get_default_configs() -> Tuple[List[Dict], int]:
    """Return (configs, full_macs), computing once and caching."""
    global _DEFAULT_CONFIGS, _FULL_MACS
    if not _DEFAULT_CONFIGS:
        _DEFAULT_CONFIGS, _FULL_MACS = find_split_configs()
    return _DEFAULT_CONFIGS, _FULL_MACS


# ---------------------------------------------------------------------------
# CLI: print split configs for inspection
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfgs, full = find_split_configs()
    print(f"Full model MACs: {full/1e6:.1f}M\n")
    print(f"{'Level':>5}  {'n_blocks':>8}  {'s_d':>5}  {'s_w':>5}  "
          f"{'MACs(M)':>8}  {'r_actual':>8}  {'imbalance':>9}")
    print("-" * 58)
    for c in cfgs:
        imb = abs(c['s_d'] - c['s_w'])
        ec  = exit_channels(c['n_blocks'], c['s_w'])
        print(f"{c['level']:>5}  {c['n_blocks']:>8}  {c['s_d']:>5.3f}  "
              f"{c['s_w']:>5.2f}  {c['macs']/1e6:>8.1f}  "
              f"{c['r_actual']:>8.3f}  {imb:>9.3f}  (exit_ch={ec})")
