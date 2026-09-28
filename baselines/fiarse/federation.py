"""
FIARSE Federation: parameter distribution and aggregation.

Core idea (from FIARSE paper Section 3):
  1. Server deep-copies the global model, generates a TopK mask at the client's
     model_size, and sends the masked model to the client.
  2. Client trains locally with standard SGD; the Bern TCB-GD function in the
     forward pass implicitly maintains the mask structure.
  3. Client returns delta = (cache_params - trained_params) for ALL parameters.
     Masked-out parameters will have zero delta (they were zeroed and received
     no gradient through Bern's biased backward).
  4. Server aggregates: for each parameter position, average over non-zero
     deltas only (partial averaging). Then subtract from global model:
     θ_global -= lr_global × avg_delta

This partial-averaging strategy means small-model clients only contribute to
parameters they used, while large-model clients contribute to more positions.

Matches Supplementary Sec. E.1, "Baseline adaptations and assignment": FIARSE
retains its magnitude-based unstructured masks, threshold-controlled biased
gradient descent, and sparse-delta aggregation.
"""

import copy
import gc

import torch
from typing import Dict, List, Optional

from .model import FIARSEResNet


class FIARSEFederation:
    """
    Manages FIARSE parameter distribution and aggregation.

    Usage:
        fed = FIARSEFederation(global_model)

        # Per round:
        # 1. Create client model with mask
        client_model = fed.distribute(model_size, device)

        # 2. Client trains locally, returns delta dict
        delta = local_training(client_model, ...)

        # 3. Aggregate all client deltas
        fed.combine(delta_list, weights)
    """

    def __init__(self, global_model: FIARSEResNet, lr_global: float = 1.0):
        self.global_model = global_model
        self.lr_global = lr_global

    def distribute(self, model_size: float,
                   device: torch.device) -> FIARSEResNet:
        """
        Create a client model: deep copy of global model with mask generated
        at the given model_size.

        Args:
            model_size: fraction of parameters to keep (0.0 to 1.0)
            device: device to place the client model on

        Returns:
            Client model (FIARSEResNet) with mask applied and Bern enabled.
        """
        client_model = copy.deepcopy(self.global_model)
        client_model.to(device)
        client_model.generate_mask(model_size=model_size, topk=True, bern=True)
        return client_model

    def combine(self,
                delta_list: List[Dict[str, torch.Tensor]],
                weights: Optional[List[float]] = None):
        """
        Aggregate client deltas into the global model using partial averaging.

        For each parameter position:
          tot_grad[pos] = sum(w_i * delta_i[pos])
          tot_mask[pos] = sum(w_i * (delta_i[pos] != 0))
          avg_delta[pos] = tot_grad[pos] / tot_mask[pos]  (0 if no contributors)
        Then:
          θ_global[pos] -= lr_global × avg_delta[pos]

        BN running stats (running_mean, running_var, num_batches_tracked) are
        updated directly without lr_global scaling, following the official impl.

        Args:
            delta_list: List of delta dicts (one per client).
                        delta = {name: cache_param - trained_param}
            weights: Optional per-client weights. If None, uniform.
        """
        if not delta_list:
            return

        if weights is None:
            weights = [1.0] * len(delta_list)

        model_params = self.global_model.state_dict()
        tot_grad = {name: torch.zeros_like(param)
                    for name, param in model_params.items()}
        tot_mask = {name: torch.zeros_like(param, dtype=torch.float32)
                    for name, param in model_params.items()}

        for delta, w in zip(delta_list, weights):
            for name, grad in delta.items():
                grad = grad.to(tot_grad[name].device)
                tot_grad[name] += w * grad
                tot_mask[name] += w * (grad != 0).float()

        for name in model_params.keys():
            if torch.any(torch.isnan(tot_grad[name])):
                raise ValueError(f"NaN in gradient for {name}. "
                                 f"Training diverged.")

            # Partial average: divide by count of non-zero contributions
            avg_grad = tot_grad[name] / tot_mask[name]
            avg_grad = torch.nan_to_num(avg_grad, nan=0.0,
                                         posinf=0.0, neginf=0.0)

            if ('running_mean' in name) or ('running_var' in name) \
                    or ('num_batches_tracked' in name):
                # BN stats: direct update (no lr_global)
                model_params[name] = model_params[name] - avg_grad
            else:
                model_params[name] -= self.lr_global * avg_grad

        self.global_model.load_state_dict(model_params)
