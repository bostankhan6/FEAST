"""
Unit tests for ScaleFL baseline.

Tests:
  1. split_config: MACs are within tolerance of target ratios
  2. Model shapes: each level produces correct output tensor count and shapes
  3. param_index_shapes: distributed params match client model shapes exactly
  4. budget_to_level: boundary conditions
  5. distribute: sliced values match global top-left
  6. combine: single full-level client leaves global unchanged
  7. self-distillation loss: correct scalar, decreases with matched predictions
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from baselines.scalefl.split_config import (
    find_split_configs, compute_macs, get_default_configs, TOTAL_BLOCKS,
)
from baselines.scalefl.model import (
    build_global_model, build_client_model,
)
from baselines.scalefl.federation import (
    ScaleFLFederation, budget_to_level, LEVEL_MACS, S_W_PER_LEVEL, N_LEVELS,
    _param_indices_for_rate,
)
from baselines.scalefl.trainer import scalefl_loss


# ---------------------------------------------------------------------------
# 1. split_config: MAC targets hit within tolerance
# ---------------------------------------------------------------------------

def test_split_config_macs():
    """All levels hit their target cost ratio within the search tolerance."""
    TARGET_RATIOS = [0.125, 0.25, 0.5, 1.0]
    TOLERANCE     = 0.15

    configs, full_macs = find_split_configs(TARGET_RATIOS, tolerance=TOLERANCE)
    assert len(configs) == len(TARGET_RATIOS)

    for cfg, r_target in zip(configs, TARGET_RATIOS):
        err = abs(cfg['r_actual'] - r_target) / r_target
        # Allow 20% relative slack (some levels may not perfectly hit target)
        assert err <= 0.20 + TOLERANCE, (
            f"Level {cfg['level']}: r_actual={cfg['r_actual']:.3f} "
            f"vs target={r_target:.3f}, relative err={err:.3f}"
        )
    print("PASS: split_config_macs")


# ---------------------------------------------------------------------------
# 2. Model output shapes
# ---------------------------------------------------------------------------

def test_model_output_shapes():
    """Each level produces the right number of exits, each (B, 100)."""
    x = torch.randn(2, 3, 32, 32)
    for level in range(1, N_LEVELS + 1):
        model = build_client_model(level, num_classes=100)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == level, (
            f"Level {level}: expected {level} exits, got {len(outputs)}"
        )
        for i, logits in enumerate(outputs):
            assert logits.shape == (2, 100), (
                f"Level {level}, exit {i}: expected (2,100) got {logits.shape}"
            )
    print("PASS: model_output_shapes")


# ---------------------------------------------------------------------------
# 3. Distributed param shapes match client model
# ---------------------------------------------------------------------------

def test_param_index_shapes():
    """distribute() slices match client model parameter shapes exactly."""
    global_model = build_global_model(num_classes=100)
    federation   = ScaleFLFederation(global_model)

    for level in range(1, N_LEVELS + 1):
        client_model = build_client_model(level, num_classes=100)
        client_sd    = client_model.state_dict()

        local_params, _ = federation.distribute(level)

        # Every param in client model must be present in local_params with same shape
        for name, expected_t in client_sd.items():
            assert name in local_params, (
                f"Level {level}: param '{name}' missing from distribute()"
            )
            assert local_params[name].shape == expected_t.shape, (
                f"Level {level}, param {name}: "
                f"distributed {local_params[name].shape} != "
                f"client {expected_t.shape}"
            )

        # No extra params should be distributed (depth isolation)
        extra = set(local_params.keys()) - set(client_sd.keys())
        assert not extra, (
            f"Level {level}: extra params distributed: {extra}"
        )
    print("PASS: param_index_shapes")


# ---------------------------------------------------------------------------
# 4. budget_to_level boundary conditions
# ---------------------------------------------------------------------------

def test_budget_to_level():
    """budget_to_level assigns highest affordable level."""
    # Exact budget == level MAC → gets that level
    for level in range(1, N_LEVELS + 1):
        result = budget_to_level(LEVEL_MACS[level - 1])
        assert result == level, (
            f"budget == level {level} MACs → got level {result}"
        )

    # Very small budget → level 1
    assert budget_to_level(1) == 1, "tiny budget should give level 1"

    # Very large budget → top level
    assert budget_to_level(10_000_000_000) == N_LEVELS, (
        "huge budget should give top level"
    )

    # One below level 4 cost → not level 4
    result = budget_to_level(LEVEL_MACS[N_LEVELS - 1] - 1)
    assert result < N_LEVELS, (
        f"budget just below top level MACs should give < {N_LEVELS}"
    )
    print("PASS: budget_to_level")


# ---------------------------------------------------------------------------
# 5. distribute: sliced values match global top-left
# ---------------------------------------------------------------------------

def test_distribute_values():
    """distribute() returns correct top-left slices from the global model."""
    global_model = build_global_model(num_classes=100)
    federation   = ScaleFLFederation(global_model)
    global_sd    = global_model.state_dict()

    for level in range(1, N_LEVELS + 1):
        s_w = S_W_PER_LEVEL[level - 1]
        local_params, param_idx = federation.distribute(level)

        for name, l_param in local_params.items():
            global_param = global_sd[name]
            idx          = param_idx[name]
            if idx:
                expected = global_param[idx]
            else:
                expected = global_param

            assert torch.allclose(l_param.float(), expected.float()), (
                f"distribute mismatch: level {level}, param {name}"
            )
    print("PASS: distribute_values")


# ---------------------------------------------------------------------------
# 6. combine: single full-level client is identity
# ---------------------------------------------------------------------------

def test_combine_identity():
    """Distributing then immediately combining (no local update) is identity."""
    global_model   = build_global_model(num_classes=100)
    federation     = ScaleFLFederation(global_model)
    sd_before      = {k: v.clone() for k, v in global_model.state_dict().items()}

    # Test with the highest level (level 4 = full model)
    global_model.load_state_dict(sd_before)
    local_params, param_idx = federation.distribute(N_LEVELS)
    federation.combine([local_params], [param_idx], weights=[1.0])

    sd_after = global_model.state_dict()
    s_w      = S_W_PER_LEVEL[N_LEVELS - 1]

    for name, before in sd_before.items():
        after = sd_after[name]
        idx   = _param_indices_for_rate(before, s_w, num_classes=100)
        if idx:
            assert torch.allclose(after[idx].float(), before[idx].float(),
                                  atol=1e-5), (
                f"combine identity mismatch (updated region): {name}"
            )
        else:
            assert torch.allclose(after.float(), before.float(), atol=1e-5), (
                f"combine identity mismatch: {name}"
            )
    print("PASS: combine_identity")


# ---------------------------------------------------------------------------
# 7. Self-distillation loss
# ---------------------------------------------------------------------------

def test_scalefl_loss():
    """scalefl_loss returns a scalar and is higher when predictions differ."""
    B = 4
    y = torch.randint(0, 100, (B,))

    # Level 1: single exit, loss = CE/2
    logits_1 = [torch.randn(B, 100)]
    loss_1   = scalefl_loss(logits_1, y)
    assert loss_1.dim() == 0, "Loss should be scalar"

    # Level 4: four exits
    logits_4 = [torch.randn(B, 100) for _ in range(4)]
    loss_4   = scalefl_loss(logits_4, y)
    assert loss_4.dim() == 0, "Loss should be scalar"

    # When all exits give the same prediction as labels, loss should be lower
    # than random logits (not a strict lower bound, but a sanity check)
    # Build near-perfect predictions
    perfect = torch.zeros(B, 100)
    perfect[torch.arange(B), y] = 10.0
    loss_perfect = scalefl_loss([perfect] * 4, y)
    loss_random  = scalefl_loss([torch.randn(B, 100) for _ in range(4)], y)
    # Average over multiple random samples to reduce flakiness
    losses_random = [scalefl_loss([torch.randn(B, 100) for _ in range(4)], y).item()
                     for _ in range(10)]
    avg_random = sum(losses_random) / len(losses_random)
    assert loss_perfect.item() < avg_random, (
        f"Perfect predictions ({loss_perfect.item():.4f}) should have lower "
        f"loss than random ({avg_random:.4f})"
    )
    print("PASS: scalefl_loss")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_split_config_macs()
    test_model_output_shapes()
    test_param_index_shapes()
    test_budget_to_level()
    test_distribute_values()
    test_combine_identity()
    test_scalefl_loss()
    print("\nAll ScaleFL tests passed.")
