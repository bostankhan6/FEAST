"""
Unit tests for FIARSE baseline.

FIARSE uses unstructured magnitude-based masking (a single global |weight|
threshold, shared across all masked layers) rather than HeteroFL/ScaleFL's
structured channel/block slicing, so its correctness properties are
different and tested accordingly:

Tests:
  1. generate_mask(model_size): actual kept-fraction matches the request
  2. Forward pass: correct output shape across a range of model sizes
  3. budget_to_model_size: boundary conditions (clip to [1/256, 1.0])
  4. Mask nesting: smaller model_size => higher (or equal) threshold, so
     its active parameter set is a subset of any larger model_size's
     (TopK-by-magnitude selection is monotonic for a fixed weight snapshot)
  5. distribute(): returns an independent deep copy, not a view into the
     global model
  6. combine(): a single client with zero delta leaves the global model
     unchanged (partial-average identity, mirroring HeteroFL's equivalent)
  7. combine(): multi-client partial averaging matches the documented
     formula (average delta over non-zero contributors only, at each
     position independently)
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

from baselines.fiarse.model import (
    build_global_model, budget_to_model_size, FULL_MODEL_MACS,
)
from baselines.fiarse.federation import FIARSEFederation

MODEL_SIZES = [1.0 / 256, 0.1, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# 1. generate_mask: actual kept-fraction matches request
# ---------------------------------------------------------------------------

def test_mask_generation_size():
    """generate_mask(model_size) keeps approximately that fraction of params."""
    model = build_global_model(num_classes=100)
    for size in MODEL_SIZES:
        actual = model.generate_mask(model_size=size, topk=True, bern=True)
        assert abs(actual - size) < 0.01, (
            f"model_size={size}: actual kept fraction {actual:.4f} too far off"
        )
    print("PASS: mask_generation_size")


# ---------------------------------------------------------------------------
# 2. Forward pass shape
# ---------------------------------------------------------------------------

def test_forward_shape():
    """All model sizes produce (batch, 100) output."""
    x = torch.randn(4, 3, 32, 32)
    model = build_global_model(num_classes=100)
    model.eval()
    for size in MODEL_SIZES:
        model.generate_mask(model_size=size, topk=True, bern=True)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 100), (
            f"model_size={size}: expected (4,100) got {out.shape}"
        )
    print("PASS: forward_shape")


# ---------------------------------------------------------------------------
# 3. budget_to_model_size boundary conditions
# ---------------------------------------------------------------------------

def test_budget_to_model_size():
    """budget_to_model_size clips to [1/256, 1.0] and is linear in between."""
    assert budget_to_model_size(0) == 1.0 / 256, "zero budget should clip to the floor"
    assert budget_to_model_size(-1) == 1.0 / 256, "negative budget should clip to the floor"
    assert budget_to_model_size(FULL_MODEL_MACS) == 1.0, "full-MAC budget should give model_size=1.0"
    assert budget_to_model_size(FULL_MODEL_MACS * 10) == 1.0, "over-budget should clip to 1.0"

    half = budget_to_model_size(FULL_MODEL_MACS / 2)
    assert abs(half - 0.5) < 1e-9, f"half-budget should give model_size=0.5, got {half}"
    print("PASS: budget_to_model_size")


# ---------------------------------------------------------------------------
# 4. Mask nesting: smaller model_size's active set subset of larger's
# ---------------------------------------------------------------------------

def test_mask_nesting():
    """For a fixed weight snapshot, a smaller model_size's TopK threshold is
    >= a larger model_size's -- so its active-parameter set nests inside the
    larger one's. This is the correctness property unstructured unstructured
    masking depends on (no re-scoring between sizes)."""
    model = build_global_model(num_classes=100)
    thresholds = {}
    for size in [0.1, 0.25, 0.5, 0.75, 1.0]:
        model.generate_mask(model_size=size, topk=True, bern=True)
        thresholds[size] = model.threshold.item()

    sizes_sorted = sorted(thresholds)
    for smaller, larger in zip(sizes_sorted, sizes_sorted[1:]):
        assert thresholds[smaller] >= thresholds[larger], (
            f"threshold at model_size={smaller} ({thresholds[smaller]:.6f}) should be "
            f">= threshold at model_size={larger} ({thresholds[larger]:.6f})"
        )
    print("PASS: mask_nesting")


# ---------------------------------------------------------------------------
# 5. distribute(): independent deep copy
# ---------------------------------------------------------------------------

def test_distribute_independent_copy():
    """distribute() must not return a view into the global model."""
    global_model = build_global_model(num_classes=100)
    federation = FIARSEFederation(global_model)
    device = torch.device('cpu')

    client_model = federation.distribute(model_size=0.5, device=device)
    assert client_model is not global_model

    global_before = {k: v.clone() for k, v in global_model.state_dict().items()}
    with torch.no_grad():
        for p in client_model.parameters():
            p.add_(1.0)

    for name, before in global_before.items():
        after = global_model.state_dict()[name]
        assert torch.equal(before, after), (
            f"Mutating client model affected global model param {name}"
        )
    print("PASS: distribute_independent_copy")


# ---------------------------------------------------------------------------
# 6. combine(): zero delta from a single client is a no-op
# ---------------------------------------------------------------------------

def test_combine_single_client_identity():
    """Combining a single client's all-zero delta must leave the global
    model completely unchanged (0/0 -> nan_to_num(0), no update)."""
    global_model = build_global_model(num_classes=100)
    federation = FIARSEFederation(global_model)
    before = {k: v.clone() for k, v in global_model.state_dict().items()}

    delta = {name: torch.zeros_like(param) for name, param in before.items()}
    federation.combine([delta], weights=[1.0])

    after = global_model.state_dict()
    for name, b in before.items():
        assert torch.equal(b, after[name]), f"Zero-delta combine changed param {name}"
    print("PASS: combine_single_client_identity")


# ---------------------------------------------------------------------------
# 7. combine(): multi-client partial averaging matches the documented formula
# ---------------------------------------------------------------------------

def test_combine_multi_client_partial_average():
    """Two clients contribute non-zero deltas to disjoint/overlapping regions
    of one parameter tensor. combine() must average only over the clients
    that actually contributed (delta != 0) at each position, not over all
    clients uniformly."""
    global_model = build_global_model(num_classes=100)
    for p in global_model.parameters():
        torch.nn.init.constant_(p, 1.0)
    federation = FIARSEFederation(global_model, lr_global=1.0)

    name = next(iter(global_model.state_dict().keys()))
    shape = global_model.state_dict()[name].shape

    delta_a = torch.zeros(shape)
    delta_b = torch.zeros(shape)
    # Position 0: only client A contributes (delta=4.0)
    delta_a.view(-1)[0] = 4.0
    # Position 1: both A and B contribute (2.0 and 6.0 -> avg 4.0)
    delta_a.view(-1)[1] = 2.0
    delta_b.view(-1)[1] = 6.0
    # Position 2: only client B contributes (delta=10.0)
    delta_b.view(-1)[2] = 10.0

    delta_list = [{name: delta_a}, {name: delta_b}]
    federation.combine(delta_list, weights=[1.0, 1.0])

    after = global_model.state_dict()[name].view(-1)
    # theta_global -= lr_global * avg_delta, starting from 1.0
    assert abs(after[0].item() - (1.0 - 4.0)) < 1e-5, "position 0 should use only client A's delta"
    assert abs(after[1].item() - (1.0 - 4.0)) < 1e-5, "position 1 should average both clients' deltas"
    assert abs(after[2].item() - (1.0 - 10.0)) < 1e-5, "position 2 should use only client B's delta"
    print("PASS: combine_multi_client_partial_average")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_mask_generation_size()
    test_forward_shape()
    test_budget_to_model_size()
    test_mask_nesting()
    test_distribute_independent_copy()
    test_combine_single_client_identity()
    test_combine_multi_client_partial_average()
    print("\nAll FIARSE tests passed.")
