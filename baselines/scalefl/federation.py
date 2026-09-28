"""
ScaleFL Federation: parameter distribution and aggregation.

Extends HeteroFL's top-left slice mechanism with depth (s_d) splitting:
  - Each complexity level has a fixed (n_blocks, s_w) pair.
  - distribute(level): extracts the top-left sub-tensor for every parameter
    that belongs to levels 1..level.  Parameters for deeper levels are not
    sent to shallower clients.
  - combine(): accumulates updates using a count tensor (identical logic to
    HeteroFL), but each client only contributes to its own depth+width slice.
    Depth masking is implicit: clients don't return params they don't have.

Aggregation equation (from ScaleFL paper Eq. 2 — same as HeteroFL Eq. 2):
  W[Z_l - Z_{l-1}] ← (1/|S^l_t|) Σ_{k in S^l_t} W_k[Z_l - Z_{l-1}]

where Z_l is the index matrix for level l (selects the top-left slice).

Parameter name conventions (must match model.py):
  stem.*                 — present in all levels
  blocks.{i}.*           — present in levels l where n_blocks[l-1] > i
  exit_classifiers.{j}.* — present in levels l >= j+1
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .model import ScaleFLResNet, build_global_model, build_client_model
from .split_config import get_default_configs, TOTAL_BLOCKS


# ---------------------------------------------------------------------------
# Level metadata
# ---------------------------------------------------------------------------

def _build_level_meta() -> Tuple[List[int], List[float], List[int], int]:
    """
    Returns (n_blocks_per_level, s_w_per_level, level_macs, full_macs).
    All lists are 0-indexed (index 0 = level 1).
    """
    configs, full_macs = get_default_configs()
    n_blocks = [c['n_blocks'] for c in configs]
    s_w      = [c['s_w']      for c in configs]
    macs     = [c['macs']     for c in configs]
    return n_blocks, s_w, macs, full_macs


N_BLOCKS_PER_LEVEL, S_W_PER_LEVEL, LEVEL_MACS, FULL_MACS = _build_level_meta()
N_LEVELS = len(N_BLOCKS_PER_LEVEL)


def budget_to_level(budget_macs: float) -> int:
    """
    Assign a client to the highest ScaleFL level affordable within budget.

    Levels are iterated highest-to-lowest. The first whose MAC cost fits
    the budget is chosen. Falls back to level 1 if nothing fits.

    Args:
        budget_macs: Client MAC budget (absolute).

    Returns:
        Level integer in {1, ..., N_LEVELS}.
    """
    for level in range(N_LEVELS, 0, -1):
        if LEVEL_MACS[level - 1] <= budget_macs:
            return level
    return 1   # fallback


# ---------------------------------------------------------------------------
# Parameter slicing utilities (same logic as HeteroFL)
# ---------------------------------------------------------------------------

def _param_indices_for_rate(param: torch.Tensor, rate: float,
                             num_classes: int = 0) -> Tuple:
    """
    Compute top-left slice indices for a given width rate.

    Conventions (same as HeteroFL federation.py):
    - 4-D conv:  (slice(n_out), slice(n_in), full, full)
                 n_in always 3 when param.shape[1] == 3 (RGB input)
    - 2-D linear: (slice(None), slice(n_in))  — output dim never sliced
                  (all levels predict full num_classes)
    - 1-D: (slice(n),) except classifier bias where shape[0] == num_classes
    - 0-D: ()
    """
    shape = param.shape
    ndim  = len(shape)

    if ndim == 0:
        return ()
    elif ndim == 1:
        if num_classes > 0 and shape[0] == num_classes:
            return (slice(None),)   # classifier bias — never scale
        n = max(1, math.ceil(rate * shape[0]))
        return (slice(0, n),)
    elif ndim == 2:
        # Linear weight: (out_features, in_features)
        # out_features = num_classes → not scaled
        n_in = max(1, math.ceil(rate * shape[1]))
        return (slice(None), slice(0, n_in))
    elif ndim == 4:
        n_out = max(1, math.ceil(rate * shape[0]))
        n_in  = 3 if shape[1] == 3 else max(1, math.ceil(rate * shape[1]))
        return (slice(0, n_out), slice(0, n_in), slice(None), slice(None))
    else:
        return tuple(slice(None) for _ in shape)


def _min_level_for_param(name: str, n_blocks_per_level: List[int]) -> int:
    """
    Return the minimum level required for a parameter to be present.

    Rules:
      stem.*                  → 1 (all levels)
      blocks.{i}.*            → smallest l s.t. n_blocks_per_level[l-1] > i
      exit_classifiers.{j}.* → j + 1
    """
    if name.startswith('stem'):
        return 1
    if name.startswith('blocks.'):
        blk_idx = int(name.split('.')[1])
        for l in range(1, len(n_blocks_per_level) + 1):
            if n_blocks_per_level[l - 1] > blk_idx:
                return l
        return len(n_blocks_per_level)
    if name.startswith('exit_classifiers.'):
        exit_idx = int(name.split('.')[1])
        return exit_idx + 1
    return 1   # fallback


# ---------------------------------------------------------------------------
# Federation class
# ---------------------------------------------------------------------------

class ScaleFLFederation:
    """
    Manages parameter distribution and aggregation for ScaleFL.

    Usage:
        fed = ScaleFLFederation(global_model)

        # Per round:
        local_params, param_idx = fed.distribute(level)
        # ... client trains, returns updated local_params ...
        fed.combine(local_params_list, param_idx_list, level_list, weights)
    """

    def __init__(self, global_model: ScaleFLResNet):
        self.global_model      = global_model
        self.num_classes       = global_model.num_classes
        self.n_blocks_per_level = N_BLOCKS_PER_LEVEL

    # ------------------------------------------------------------------
    # distribute
    # ------------------------------------------------------------------

    def distribute(self, level: int) -> Tuple[Dict[str, torch.Tensor],
                                               Dict[str, Tuple]]:
        """
        Extract parameter slice for a client at the given level.

        Returns:
            local_params: {name → tensor} — only params for this level
            param_idx:    {name → slice tuple} — width indices used
        """
        s_w      = S_W_PER_LEVEL[level - 1]
        global_sd = self.global_model.state_dict()

        local_params = {}
        param_idx    = {}

        for name, param in global_sd.items():
            if _min_level_for_param(name, self.n_blocks_per_level) > level:
                continue   # this param doesn't exist in the client model
            idx = _param_indices_for_rate(param, s_w, self.num_classes)
            local_params[name] = param[idx].clone() if idx else param.clone()
            param_idx[name]    = idx

        return local_params, param_idx

    # ------------------------------------------------------------------
    # combine
    # ------------------------------------------------------------------

    def combine(self,
                local_params_list: List[Dict[str, torch.Tensor]],
                param_idx_list:    List[Dict[str, Tuple]],
                weights:           Optional[List[float]] = None):
        """
        Aggregate client updates into the global model.

        Depth masking is implicit: each client's local_params only contains
        parameters for their level, so deeper params are naturally untouched
        by shallow clients.  Width masking uses the same count-tensor logic
        as HeteroFL.

        Args:
            local_params_list: One dict per client.
            param_idx_list:    One dict per client (name → slice tuple).
            weights:           Per-client sample counts.  None = uniform.
        """
        if weights is None:
            weights = [1.0] * len(local_params_list)

        global_sd = self.global_model.state_dict()

        tmp_v = {n: torch.zeros_like(p) for n, p in global_sd.items()}
        count = {n: torch.zeros_like(p, dtype=torch.float32)
                 for n, p in global_sd.items()}

        # Accumulate raw weighted sums (do NOT pre-divide by total_weight).
        # Each parameter position is normalised by the sum of weights of the
        # clients that actually contributed to it — not by the global total.
        # Pre-dividing by total_weight is wrong when depth-masking means only
        # a subset of clients update deeper parameters: it would dampen those
        # updates by (subset_weight / total_weight) < 1, collapsing the deeper
        # layers toward zero on every round they are touched.
        for local_params, param_idx, w in zip(local_params_list,
                                               param_idx_list, weights):
            for name, l_param in local_params.items():
                idx     = param_idx[name]
                l_param = l_param.float().to(tmp_v[name].device)
                if idx:
                    tmp_v[name][idx] += w * l_param
                    count[name][idx] += w
                else:
                    tmp_v[name]      += w * l_param
                    count[name]      += w

        # Normalise each position by its own accumulated weight.
        # Positions with no updates keep their current global value.
        new_sd = {}
        for name, param in global_sd.items():
            updated      = tmp_v[name].clone()
            updated_mask = count[name] > 0
            updated[updated_mask] /= count[name][updated_mask]
            updated[~updated_mask] = param.float()[~updated_mask]
            new_sd[name] = updated.to(param.dtype)

        self.global_model.load_state_dict(new_sd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_level_macs(self, level: int) -> int:
        return LEVEL_MACS[level - 1]

    def get_level_s_w(self, level: int) -> float:
        return S_W_PER_LEVEL[level - 1]

    @staticmethod
    def assign_level(budget_macs: float) -> int:
        return budget_to_level(budget_macs)
