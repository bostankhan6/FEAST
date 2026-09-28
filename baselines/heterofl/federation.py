"""
HeteroFL Federation: parameter distribution and aggregation.

Faithful re-implementation of the split_model / distribute / combine logic
from the official HeteroFL repository (fed.py).

Core idea: the global model holds the full-width parameters. Each client
receives a contiguous top-left slice of every weight tensor (first K output
channels and first K input channels). On aggregation, contributions are
accumulated into those same slices and divided by a count tensor.

Budget → tier assignment: MACs-based (highest affordable tier wins).
Tier MACs scale as rate² × base_macs (width scales both dim of each conv).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .model import HeteroFLResNet, build_global_model, build_heterofl_model


# ---------------------------------------------------------------------------
# Tier definitions (faithful to HeteroFL paper Table 1)
# ---------------------------------------------------------------------------

# Fixed model rates from the HeteroFL paper. Matches Supplementary Sec. E.1,
# "Baseline adaptations and assignment": width multipliers {1/16, 1/8, 1/4,
# 1/2, 1}, top-left channel slicing, and position-wise aggregation.
TIER_RATES = {
    'a': 1.0,
    'b': 0.5,
    'c': 0.25,
    'd': 0.125,
    'e': 0.0625,
}

# Full-width (~rate=1.0) approximate MACs for CIFAR-100 32×32 input.
# Computed analytically: each stage contributes ~137M MACs at rate=1.0
# (see model.py docstring). Total ≈ 550M MACs.
FULL_MODEL_MACS = 550_000_000  # ~550M

# MACs for each tier (scales as rate² since both width dims shrink)
TIER_MACS = {
    name: int(FULL_MODEL_MACS * rate ** 2)
    for name, rate in TIER_RATES.items()
}


def budget_to_tier(budget_macs: float) -> str:
    """
    Assign a client to the highest HeteroFL tier affordable within budget.

    Tiers are sorted highest-to-lowest. The first (largest) tier whose
    MAC cost fits the client's budget is chosen. If even the smallest
    tier (e) doesn't fit, tier 'e' is returned as a fallback
    (implementation note: tier 'e' at ~2.1M always fits realistic budgets).

    Args:
        budget_macs: Client's MAC budget (absolute, e.g. 137_000_000).

    Returns:
        Tier name string in {'a', 'b', 'c', 'd', 'e'}.
    """
    for tier_name in ['a', 'b', 'c', 'd', 'e']:
        if TIER_MACS[tier_name] <= budget_macs:
            return tier_name
    return 'e'  # fallback


# ---------------------------------------------------------------------------
# Parameter index computation
# ---------------------------------------------------------------------------

def _param_indices_for_rate(global_param: torch.Tensor,
                             rate: float,
                             num_classes: int = 0) -> Tuple:
    """
    Compute the slice indices that select the client's sub-tensor from
    the global parameter tensor.

    Convention (following HeteroFL fed.py):
    - For 4D conv weights (out_ch, in_ch, kH, kW): slice first two dims.
      The RGB input (in_ch == 3) is never scaled.
    - For 2D linear weights (out, in): only scale in_features.
      out_features (num_classes) is never scaled.
    - For 1D tensors (BN weight/bias, scaler, classifier bias):
      scale unless size == num_classes (classifier bias is never scaled).
    - Scaler parameters are 1D (1,): always take full slice.

    Args:
        global_param: The global model's parameter tensor.
        rate:         Client model rate in (0, 1].
        num_classes:  Number of output classes; used to detect classifier
                      parameters that must not be scaled.

    Returns a tuple of slices, one per dimension.
    """
    shape = global_param.shape
    ndim  = len(shape)

    if ndim == 0:
        return ()
    elif ndim == 1:
        # Classifier bias has shape (num_classes,) — never scale
        if num_classes > 0 and shape[0] == num_classes:
            return (slice(None),)
        # BN weight/bias and Scaler — scale first dim
        n = max(1, math.ceil(rate * shape[0]))
        return (slice(0, n),)
    elif ndim == 2:
        # Linear weight: (out_features, in_features).
        # For the classifier, out_features == num_classes and must not be scaled
        # (all tiers predict the full label set).  In_features come from the
        # last hidden stage and should be scaled by rate.
        # This architecture has no hidden-to-hidden FC layers, so it is safe to
        # always keep n_out at full size and only scale n_in.
        n_in = max(1, math.ceil(rate * shape[1]))
        return (slice(None), slice(0, n_in))
    elif ndim == 4:
        # Conv weight: (out_ch, in_ch, kH, kW) — kH/kW always full
        n_out = max(1, math.ceil(rate * shape[0]))
        # Input channels: only scale hidden dims, never the original RGB input (3)
        if shape[1] == 3:
            n_in = 3
        else:
            n_in = max(1, math.ceil(rate * shape[1]))
        return (slice(0, n_out), slice(0, n_in),
                slice(None),     slice(None))
    else:
        # Unexpected shape: return full tensor
        return tuple(slice(None) for _ in shape)


# ---------------------------------------------------------------------------
# Federation class
# ---------------------------------------------------------------------------

class HeteroFLFederation:
    """
    Manages parameter distribution and aggregation for HeteroFL.

    Usage:
        fed = HeteroFLFederation(global_model, num_classes=100)

        # --- Per round ---
        # 1. Distribute to clients
        client_params, param_idx = fed.distribute(tier_name)

        # 2. Client trains locally, returns updated state_dict
        ...

        # 3. Combine all clients' updates into global model
        fed.combine(list_of_local_params, list_of_tier_names)
    """

    def __init__(self, global_model: HeteroFLResNet):
        self.global_model = global_model
        self.num_classes   = global_model.classifier.out_features
        # Cache param names for indexing
        self.param_names = [n for n, _ in global_model.named_parameters()]

    # ------------------------------------------------------------------
    # distribute: extract client sub-model from global model
    # ------------------------------------------------------------------

    def distribute(self, tier_name: str) -> Tuple[Dict[str, torch.Tensor],
                                                   Dict[str, Tuple]]:
        """
        Extract the parameter slice for a client at the given tier.

        Returns:
            local_params: dict {param_name → tensor} — client's parameters
            param_idx:    dict {param_name → tuple_of_slices} — indices used
        """
        rate       = TIER_RATES[tier_name]
        global_sd  = self.global_model.state_dict()

        local_params = {}
        param_idx    = {}

        for name, param in global_sd.items():
            idx = _param_indices_for_rate(param, rate, self.num_classes)
            local_params[name] = param[idx].clone() if idx else param.clone()
            param_idx[name]    = idx

        return local_params, param_idx

    # ------------------------------------------------------------------
    # combine: aggregate client updates into global model
    # ------------------------------------------------------------------

    def combine(self,
                local_params_list: List[Dict[str, torch.Tensor]],
                param_idx_list:    List[Dict[str, Tuple]],
                weights:           Optional[List[float]] = None):
        """
        Aggregate heterogeneous client updates into the global model.

        Each client's parameters slot into the top-left of the global tensor.
        A count tensor tracks how many clients contributed to each element,
        allowing correct averaging even when clients have different widths.

        Args:
            local_params_list: One dict per client (name → updated tensor).
            param_idx_list:    One dict per client (name → slice tuple).
            weights:           Optional per-client sample counts for weighted avg.
                               If None, uniform averaging is used.
        """
        if weights is None:
            weights = [1.0] * len(local_params_list)

        global_sd = self.global_model.state_dict()

        # Accumulation buffers
        tmp_v  = {n: torch.zeros_like(p) for n, p in global_sd.items()}
        count  = {n: torch.zeros_like(p, dtype=torch.float32)
                  for n, p in global_sd.items()}

        # Accumulate raw weighted sums. Do NOT pre-divide by total_weight —
        # each position must be normalised by the sum of weights of clients
        # that actually contributed to it (outer channels only get tier-a
        # updates; pre-dividing by total_weight would dampen them wrongly).
        for local_params, param_idx, w in zip(local_params_list,
                                               param_idx_list, weights):
            for name in global_sd:
                idx     = param_idx[name]
                l_param = local_params[name].float().to(tmp_v[name].device)
                if idx:
                    tmp_v[name][idx]  += w * l_param
                    count[name][idx]  += w
                else:
                    tmp_v[name]       += w * l_param
                    count[name]       += w

        # Normalise each position by its own accumulated weight.
        # Positions with no updates keep old global value.
        new_sd = {}
        for name, param in global_sd.items():
            updated      = tmp_v[name].clone()
            updated_mask = count[name] > 0
            updated[updated_mask]  /= count[name][updated_mask]
            updated[~updated_mask]  = param.float()[~updated_mask]
            new_sd[name] = updated.to(param.dtype)

        self.global_model.load_state_dict(new_sd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_rate(self, tier_name: str) -> float:
        return TIER_RATES[tier_name]

    def get_tier_macs(self, tier_name: str) -> int:
        return TIER_MACS[tier_name]

    @staticmethod
    def assign_tier(budget_macs: float) -> str:
        return budget_to_tier(budget_macs)
