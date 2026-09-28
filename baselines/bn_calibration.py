"""
BN calibration for models trained with track_running_stats=False.

Standard .train() forward passes are no-ops for such models: there is no buffer
to store statistics into, so .eval() still uses per-batch statistics and accuracy
collapses on non-representative (e.g. single-class) batches.

calibrate_bn_running_stats() temporarily enables running-stat tracking with
momentum=None (cumulative moving average), runs a calibration pass, and leaves
the model with frozen global statistics for .eval().  This mirrors what FEAST
receives via set_running_statistics(), making baseline evaluation fully
batch-order-independent.
"""

import torch
import torch.nn as nn


def calibrate_bn_running_stats(
    model: nn.Module,
    calibration_loader,
    device: torch.device,
) -> None:
    """
    Proper BN calibration for models with track_running_stats=False.

    For every BatchNorm2d:
      - Enables track_running_stats=True
      - Resets running_mean / running_var to canonical initial values (0 / 1)
      - Sets momentum=None  ->  cumulative average (weight = 1/n per batch)

    After a full pass over calibration_loader in train mode the BN layers hold
    the true per-channel mean and variance of the calibration set.  The caller
    must switch to model.eval() to use these frozen stats during test evaluation.

    Args:
        model:              The model to calibrate (modified in place).
        calibration_loader: DataLoader over the held-out calibration set
                            (same set used for FEAST's set_running_statistics).
        device:             Device the model is on.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            # These buffers were registered as None when track_running_stats=False
            m.running_mean = torch.zeros(m.num_features, device=device)
            m.running_var  = torch.ones(m.num_features,  device=device)
            m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            # momentum=None: effective lr = 1/num_batches_tracked  (cumulative avg)
            m.momentum = None

    model.train()
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x.to(device))
    # Caller is responsible for calling model.eval()
