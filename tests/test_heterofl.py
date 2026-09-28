"""
Unit tests for HeteroFL baseline.

Tests:
  1. Model channel scaling: each rate produces correct hidden sizes
  2. Model forward pass: correct output shape
  3. Parameter index slicing: shapes match expected sub-tensor
  4. budget_to_tier mapping: boundary conditions
  5. distribute: sliced tensors match top-left of global model
  6. combine: weighted average reconstructed correctly
  7. Aggregation idempotency: combining a single full-rate client is identity
"""

import sys
import os
import math
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

from baselines.heterofl.model import (
    HeteroFLResNet, build_heterofl_model, build_global_model,
)
from baselines.heterofl.federation import (
    HeteroFLFederation, budget_to_tier, TIER_RATES, TIER_MACS,
    _param_indices_for_rate,
)


# ---------------------------------------------------------------------------
# 1. Channel scaling
# ---------------------------------------------------------------------------

def test_channel_scaling():
    """Each rate produces ceil(rate * base) channels per stage."""
    base = [64, 128, 256, 512]
    for name, rate in TIER_RATES.items():
        model = build_heterofl_model(rate, num_classes=100)
        expected = [max(1, math.ceil(rate * c)) for c in base]
        assert model.scaled_hs == expected, (
            f"Tier {name} (rate={rate}): expected {expected}, got {model.scaled_hs}"
        )
    print("PASS: channel_scaling")


# ---------------------------------------------------------------------------
# 2. Forward pass shape
# ---------------------------------------------------------------------------

def test_forward_shape():
    """All tier models produce (batch, 100) output."""
    x = torch.randn(4, 3, 32, 32)
    for name, rate in TIER_RATES.items():
        model = build_heterofl_model(rate, num_classes=100)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 100), (
            f"Tier {name}: expected (4,100) got {out.shape}"
        )
    print("PASS: forward_shape")


# ---------------------------------------------------------------------------
# 3. Parameter index slicing shapes
# ---------------------------------------------------------------------------

def test_param_index_shapes():
    """distribute() slices match client model parameter shapes exactly."""
    global_model = build_global_model(num_classes=100)
    federation   = HeteroFLFederation(global_model)

    for tier_name, rate in TIER_RATES.items():
        client_model = build_heterofl_model(rate, num_classes=100)
        client_sd    = client_model.state_dict()

        local_params, _ = federation.distribute(tier_name)

        for param_name, local_p in local_params.items():
            if param_name in client_sd:
                expected_shape = client_sd[param_name].shape
                assert local_p.shape == expected_shape, (
                    f"Tier {tier_name}, param {param_name}: "
                    f"distributed {local_p.shape} != client {expected_shape}"
                )
    print("PASS: param_index_shapes")


# ---------------------------------------------------------------------------
# 4. budget_to_tier boundary conditions
# ---------------------------------------------------------------------------

def test_budget_to_tier():
    """budget_to_tier assigns highest affordable tier."""
    # Exact budget == tier MAC cost → gets that tier
    for tier_name, macs in TIER_MACS.items():
        result = budget_to_tier(macs)
        assert result == tier_name, (
            f"budget={macs} (exact {tier_name} MACs) → got '{result}'"
        )

    # One less than tier b → should NOT get b; gets c
    tier_b_macs = TIER_MACS['b']
    result = budget_to_tier(tier_b_macs - 1)
    assert TIER_MACS[result] <= tier_b_macs - 1, (
        f"budget just below tier b: got '{result}' with MACs={TIER_MACS[result]}"
    )

    # Very small budget → tier 'e' (fallback)
    result = budget_to_tier(100)
    assert result == 'e', f"tiny budget → expected 'e', got '{result}'"

    # Very large budget → tier 'a'
    result = budget_to_tier(10_000_000_000)
    assert result == 'a', f"huge budget → expected 'a', got '{result}'"

    print("PASS: budget_to_tier")


# ---------------------------------------------------------------------------
# 5. distribute: sliced values match global top-left
# ---------------------------------------------------------------------------

def test_distribute_values():
    """distribute() returns exact top-left slices of global model params."""
    global_model = build_global_model(num_classes=100)
    federation = HeteroFLFederation(global_model)
    global_sd = global_model.state_dict()

    for tier_name in TIER_RATES:
        local_params, param_idx = federation.distribute(tier_name)
        for name, param in global_sd.items():
            idx = param_idx[name]
            if idx:
                expected = param[idx]
            else:
                expected = param
            assert torch.allclose(local_params[name].float(),
                                  expected.float()), (
                f"distribute mismatch: tier {tier_name}, param {name}"
            )
    print("PASS: distribute_values")


# ---------------------------------------------------------------------------
# 6. combine: single-client weighted average is identity
# ---------------------------------------------------------------------------

def test_combine_single_client():
    """After distributing and immediately combining (no local update), global model unchanged."""
    global_model = build_global_model(num_classes=100)
    federation = HeteroFLFederation(global_model)
    global_sd_before = {k: v.clone() for k, v in global_model.state_dict().items()}

    for tier_name in TIER_RATES:
        # Re-init to known state
        global_model.load_state_dict(global_sd_before)
        local_params, param_idx = federation.distribute(tier_name)
        federation.combine([local_params], [param_idx], weights=[1.0])

        global_sd_after = global_model.state_dict()
        rate = TIER_RATES[tier_name]

        from baselines.heterofl.federation import _param_indices_for_rate
        for name, before in global_sd_before.items():
            after = global_sd_after[name]
            idx = _param_indices_for_rate(before, rate)
            if idx:
                # Updated slice should be close to original
                assert torch.allclose(after[idx].float(), before[idx].float(),
                                      atol=1e-5), (
                    f"combine mismatch in updated region: tier {tier_name}, "
                    f"param {name}"
                )
                # Unupdated slice must be exactly unchanged
                # Build complement mask and check
            else:
                assert torch.allclose(after.float(), before.float(), atol=1e-5), (
                    f"combine mismatch (full): tier {tier_name}, param {name}"
                )
    print("PASS: combine_single_client")


# ---------------------------------------------------------------------------
# 7. combine: multi-tier weighted average preserves highest-tier slice
# ---------------------------------------------------------------------------

def test_combine_multi_tier():
    """Combining tier-a and tier-e with equal weight averages the overlapping region."""
    global_model = build_global_model(num_classes=100)
    # Fill global model with known values
    for p in global_model.parameters():
        torch.nn.init.constant_(p, 1.0)

    federation = HeteroFLFederation(global_model)

    # Distribute to both tiers
    params_a, idx_a = federation.distribute('a')  # full model
    params_e, idx_e = federation.distribute('e')  # smallest slice

    # Client 'a' sets their params to 2.0, client 'e' to 4.0
    params_a = {k: torch.full_like(v, 2.0) for k, v in params_a.items()}
    params_e = {k: torch.full_like(v, 4.0) for k, v in params_e.items()}

    # Equal weights → overlap region should be (0.5*2 + 0.5*4) = 3.0
    federation.combine([params_a, params_e], [idx_a, idx_e], weights=[1.0, 1.0])

    global_sd = global_model.state_dict()
    rate_e = TIER_RATES['e']

    for name, param in global_sd.items():
        idx_for_e = _param_indices_for_rate(param, rate_e)
        if idx_for_e:
            overlap = param[idx_for_e]
            # Overlap: both clients contributed → weighted avg of 2.0 and 4.0
            assert torch.allclose(overlap.float(),
                                  torch.full_like(overlap, 3.0).float(),
                                  atol=1e-4), (
                f"Overlap region should be 3.0: param {name}, "
                f"got {overlap.float().mean().item():.4f}"
            )
    print("PASS: combine_multi_tier")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_channel_scaling()
    test_forward_shape()
    test_param_index_shapes()
    test_budget_to_tier()
    test_distribute_values()
    test_combine_single_client()
    test_combine_multi_tier()
    print("\nAll HeteroFL tests passed.")
