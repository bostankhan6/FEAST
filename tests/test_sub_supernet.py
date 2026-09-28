"""
Unit tests for sub-supernet functionality.

max_extra_blocks_per_stage is read from the real config
(4-stage-supernet-cifar100-v2.json) rather than hardcoded, so these tests
stay correct if that value changes.
Expansion ratios are STAGE-LEVEL: e has length num_stages=4 (not 24).

Tests:
1. GenericOFAResNet with per_position_max_d — block count, forward pass, depth validation
2. compute_sub_supernet_bounds() — bounds computed correctly from synthetic cache
3. create_sub_supernet() — per-stage depth + width reduction, param count
4. Weight copying — weights copied into correct positions
5. add_sub_supernet() — sparse aggregation correctness
6. Forward pass — min/max config through sub-supernet

Runnable directly (`python tests/test_sub_supernet.py`) or via pytest.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import copy
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from feast.Server.generic_server_model import GenericServerOFA
from feast.elastic_nn.generic_ofa_network import GenericOFAResNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..',
                           'configs', 'supernets', '4-stage-supernet-cifar100-v2.json')

# e is now stage-level: 4 values (one per stage)
NUM_STAGES = 4


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg['n_classes'] = 100
    return cfg


def make_e_values(ratio):
    """Return a list of NUM_STAGES copies of ratio — one per stage."""
    return [ratio] * NUM_STAGES


def make_synthetic_cache():
    """
    Create a tiny in-memory subnet cache.
    Each config has:
      d  — list of 4 ints in [0, 5]
      e  — list of 4 floats (one expansion ratio per stage, stage-level)
      w_indices — list of 5 ints

    Five entries cover the 50M–600M range with varying depths so that
    compute_sub_supernet_bounds() returns genuinely different per-stage bounds.
    """
    records = [
        # Very small — shallow, thin
        {'macs': 52e6,  'params': 0.5e6, 'd': str([0,0,0,0]),
         'e': str(make_e_values(0.10)),  'w_indices': str([3,0,0,0,0])},
        # Small-medium — last stage slightly deeper
        {'macs': 130e6, 'params': 1.2e6, 'd': str([0,0,0,1]),
         'e': str(make_e_values(0.14)),  'w_indices': str([5,1,1,1,1])},
        # Medium
        {'macs': 280e6, 'params': 2.8e6, 'd': str([1,1,2,3]),
         'e': str(make_e_values(0.18)),  'w_indices': str([7,2,3,4,5])},
        # Large
        {'macs': 460e6, 'params': 5.0e6, 'd': str([3,3,4,5]),
         'e': str(make_e_values(0.22)),  'w_indices': str([8,5,6,7,8])},
        # Max budget
        {'macs': 598e6, 'params': 7.0e6, 'd': str([5,5,5,5]),
         'e': str(make_e_values(0.25)),  'w_indices': str([9,7,8,8,9])},
    ]
    return pd.DataFrame(records)


def build_server(extra_cfg=None):
    """Build a GenericServerOFA backed by a temporary synthetic cache CSV."""
    cfg = load_config()
    cfg['resource_heterogeneity'] = True
    cfg['resource_distribution_type'] = 'zipf'
    cfg['resource_zipf_alpha'] = 1.2
    cfg['resource_force_max_clients'] = 2
    cfg['resource_max_mac'] = 600e6
    if extra_cfg:
        cfg.update(extra_cfg)

    # Write synthetic cache to a temp file
    df = make_synthetic_cache()
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
    df.to_csv(tmp.name, index=False)
    tmp.close()
    cfg['subnet_cache_path'] = tmp.name

    np.random.seed(0)
    return GenericServerOFA(
        arch_params=cfg,
        sampling_method='TS_optimal_path',
        num_cli_total=10,
    )


def _build_low_budget_sub_supernet(server):
    """Sub-supernet at the low-budget anchor used by tests 4-6."""
    macs = server.subnet_cache_macs
    budget_low = (macs[0] + macs[1]) / 2
    return server.create_sub_supernet(budget_low)


# ---------------------------------------------------------------------------
# Pytest fixtures (module-scoped: the server/sub-supernet are read-only in
# these tests, so building them once per module is safe and avoids redundant
# GA-cache loading + model construction per test).
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def server():
    return build_server()


@pytest.fixture(scope='module')
def _sub_supernet_and_info(server):
    return _build_low_budget_sub_supernet(server)


@pytest.fixture(scope='module')
def sub_supernet(_sub_supernet_and_info):
    return _sub_supernet_and_info[0]


@pytest.fixture(scope='module')
def sub_info(_sub_supernet_and_info):
    return _sub_supernet_and_info[1]


# ---------------------------------------------------------------------------
# Test 1: GenericOFAResNet with per_position_max_d
# ---------------------------------------------------------------------------
def test_per_position_max_d_network():
    print("\n=== Test 1: GenericOFAResNet with per_position_max_d ===")
    cfg = load_config()

    def make_net(**kwargs):
        return GenericOFAResNet(
            num_stages=cfg['num_stages'],
            initial_input_hw=cfg['initial_input_hw'],
            initial_input_channels=cfg['initial_input_channels'],
            stem_stride=cfg['stem_stride'],
            original_stem_out_channels=cfg['original_stem_out_channels'],
            original_stage_base_channels=cfg['original_stage_base_channels'],
            stage_downsample_factors=cfg['stage_downsample_factors'],
            max_extra_blocks_per_stage=cfg['max_extra_blocks_per_stage'],
            channel_divisible_by=cfg['channel_divisible_by'],
            width_multiplier_choices=cfg['width_multiplier_choices'],
            expansion_ratio_choices=cfg['expansion_ratio_choices'],
            n_classes=100,
            **kwargs
        )

    # 1a. Full supernet: 4 stages, each with (max_extra_blocks_per_stage + 1) blocks
    max_extra = cfg['max_extra_blocks_per_stage']
    full = make_net()
    assert full.max_extra_blocks_per_stage_list == [max_extra] * NUM_STAGES
    assert len(full.blocks) == (max_extra + 1) * NUM_STAGES
    assert [len(g) for g in full.grouped_block_index] == [max_extra + 1] * NUM_STAGES
    print(f"  Full supernet: {len(full.blocks)} blocks, {sum(p.numel() for p in full.parameters())/1e6:.1f}M params")

    # 1b. Sub-supernet with per_position_max_d=[0,0,0,1]: 1+1+1+2=5 blocks
    sub = make_net(per_position_max_d=[0, 0, 0, 1])
    assert sub.max_extra_blocks_per_stage_list == [0, 0, 0, 1]
    assert len(sub.blocks) == 5
    assert [len(g) for g in sub.grouped_block_index] == [1, 1, 1, 2]
    print(f"  Sub [0,0,0,1]:  {len(sub.blocks)} blocks, {sum(p.numel() for p in sub.parameters())/1e6:.1f}M params")

    # 1c. Forward pass with various d values (e_indices length = num_stages = 4)
    x = torch.randn(2, 3, 32, 32)
    e_idx = [0] * NUM_STAGES
    for d_val in [[0,0,0,0], [0,0,0,1]]:
        sub.set_active_subnet(d=d_val, e_indices=e_idx, w_indices=[0,0,0,0,0])
        y = sub(x)
        assert y.shape == (2, 100), f"Bad output shape for d={d_val}: {y.shape}"
    print(f"  Forward passes OK")

    # 1d. Depth out-of-range raises ValueError
    try:
        sub.set_active_subnet(d=[1, 0, 0, 0], e_indices=e_idx, w_indices=[0,0,0,0,0])
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print(f"  Out-of-range depth correctly raises ValueError")

    # 1e. e_indices wrong length raises ValueError (5 != 4)
    try:
        sub.set_active_subnet(d=[0,0,0,0], e_indices=[0]*5, w_indices=[0,0,0,0,0])
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print(f"  Wrong e_indices length correctly raises ValueError")

    print("PASSED: per_position_max_d network")


# ---------------------------------------------------------------------------
# Test 2: compute_sub_supernet_bounds
# ---------------------------------------------------------------------------
def test_compute_sub_supernet_bounds(server):
    print("\n=== Test 2: compute_sub_supernet_bounds ===")

    # _prepare_subnet_cache recalculates MACs at runtime via subnet_macs(),
    # ignoring the CSV 'macs' column.  Use the actual values from the server.
    cache = server.subnet_cache
    macs  = server.subnet_cache_macs   # sorted ascending
    assert len(cache) == 5, f"Expected 5 cached configs, got {len(cache)}"
    num_stages = server.arch_params['num_stages']

    print(f"  Actual cache MACs: {[f'{m/1e6:.2f}M' for m in macs]}")
    print(f"  Cache d-vectors:   {[c['d'] for c in cache]}")

    # --- Low budget: midpoint between config-0 and config-1 → only 1 valid ---
    budget_low = (macs[0] + macs[1]) / 2
    b_low = server.compute_sub_supernet_bounds(budget_low)
    print(f"  low({budget_low/1e6:.1f}M) → max_d={b_low['max_d']}, max_w={b_low['max_w_indices']}")
    assert len(b_low['valid_cache_indices']) == 1, \
        f"Expected 1 valid config, got {len(b_low['valid_cache_indices'])}"
    expected_low_d = list(cache[0]['d'])
    assert b_low['max_d'] == expected_low_d, \
        f"Expected max_d={expected_low_d}, got {b_low['max_d']}"

    # --- Medium budget: midpoint between config-2 and config-3 → first 3 valid ---
    budget_med = (macs[2] + macs[3]) / 2
    b_med = server.compute_sub_supernet_bounds(budget_med)
    print(f"  med({budget_med/1e6:.1f}M) → max_d={b_med['max_d']}, max_w={b_med['max_w_indices']}")
    assert len(b_med['valid_cache_indices']) == 3, \
        f"Expected 3 valid configs, got {len(b_med['valid_cache_indices'])}"
    expected_med_d = [max(cache[i]['d'][j] for i in range(3)) for j in range(num_stages)]
    assert b_med['max_d'] == expected_med_d, \
        f"Expected max_d={expected_med_d}, got {b_med['max_d']}"

    # --- Full budget: above all configs → all 5 valid ---
    budget_full = macs[-1] * 1.05
    b_full = server.compute_sub_supernet_bounds(budget_full)
    print(f"  full({budget_full/1e6:.1f}M) → max_d={b_full['max_d']}, max_w={b_full['max_w_indices']}")
    assert len(b_full['valid_cache_indices']) == 5, \
        f"Expected 5 valid configs, got {len(b_full['valid_cache_indices'])}"
    expected_full_d = [max(cache[i]['d'][j] for i in range(5)) for j in range(num_stages)]
    assert b_full['max_d'] == expected_full_d, \
        f"Expected max_d={expected_full_d}, got {b_full['max_d']}"

    print("PASSED: compute_sub_supernet_bounds")


# ---------------------------------------------------------------------------
# Test 3: create_sub_supernet — param reduction and per-stage structure
# ---------------------------------------------------------------------------
def test_create_sub_supernet(server):
    print("\n=== Test 3: create_sub_supernet ===")

    cache = server.subnet_cache
    macs  = server.subnet_cache_macs
    num_stages = server.arch_params['num_stages']

    global_params = sum(p.numel() for p in server.model.parameters())
    print(f"  Full supernet: {global_params/1e6:.2f}M params")

    # Low budget: midpoint between config-0 and config-1 → only config-0 valid
    budget_low = (macs[0] + macs[1]) / 2
    sub_low, info_low = server.create_sub_supernet(budget_low)
    sub_low_p = sum(p.numel() for p in sub_low.parameters())
    expected_low_d = list(cache[0]['d'])
    expected_low_blocks = [d + 1 for d in expected_low_d]
    print(f"  Sub-supernet low({budget_low/1e6:.1f}M): {sub_low_p/1e6:.2f}M params  "
          f"(reduction {global_params/sub_low_p:.2f}x), "
          f"per_stage_max_d={sub_low.max_extra_blocks_per_stage_list}, "
          f"blocks={[len(g) for g in sub_low.grouped_block_index]}")
    assert sub_low_p < global_params
    assert not info_low['is_full_supernet']
    assert sub_low.max_extra_blocks_per_stage_list == expected_low_d, \
        f"Expected {expected_low_d}, got {sub_low.max_extra_blocks_per_stage_list}"
    assert [len(g) for g in sub_low.grouped_block_index] == expected_low_blocks, \
        f"Expected {expected_low_blocks}, got {[len(g) for g in sub_low.grouped_block_index]}"

    # Medium budget: midpoint between config-2 and config-3 → first 3 valid
    budget_med = (macs[2] + macs[3]) / 2
    sub_med, info_med = server.create_sub_supernet(budget_med)
    sub_med_p = sum(p.numel() for p in sub_med.parameters())
    expected_med_d = [max(cache[i]['d'][j] for i in range(3)) for j in range(num_stages)]
    expected_med_blocks = [d + 1 for d in expected_med_d]
    print(f"  Sub-supernet med({budget_med/1e6:.1f}M): {sub_med_p/1e6:.2f}M params  "
          f"(reduction {global_params/sub_med_p:.2f}x), "
          f"per_stage_max_d={sub_med.max_extra_blocks_per_stage_list}")
    assert sub_med_p < global_params
    assert sub_med.max_extra_blocks_per_stage_list == expected_med_d, \
        f"Expected {expected_med_d}, got {sub_med.max_extra_blocks_per_stage_list}"
    assert [len(g) for g in sub_med.grouped_block_index] == expected_med_blocks, \
        f"Expected {expected_med_blocks}, got {[len(g) for g in sub_med.grouped_block_index]}"

    print("PASSED: create_sub_supernet")


# ---------------------------------------------------------------------------
# Test 4: Weight copying
# ---------------------------------------------------------------------------
def test_weight_copying(server, sub_supernet, sub_info):
    print("\n=== Test 4: weight_copying ===")

    if sub_info['is_full_supernet']:
        print("SKIPPED: full supernet")
        return

    mapping = sub_info['mapping']
    global_state = server.model.state_dict()
    sub_state = sub_supernet.state_dict()

    errors = []
    checked = 0
    for sub_key, (global_key, slice_info) in list(mapping.items())[:8]:
        stype = slice_info.get('type', 'other')
        if stype == 'conv':
            out_ch, in_ch = slice_info['out_ch'], slice_info['in_ch']
            expected = global_state[global_key][:out_ch, :in_ch, :, :]
            actual = sub_state[sub_key]
            if not torch.allclose(expected, actual):
                errors.append(f"{sub_key}: mismatch")
            else:
                checked += 1
        elif stype == 'bn':
            dim = slice_info['dim']
            expected = global_state[global_key][:dim]
            actual = sub_state[sub_key]
            if not torch.allclose(expected, actual):
                errors.append(f"{sub_key}: mismatch")
            else:
                checked += 1

    assert not errors, f"Weight copy errors: {errors}"
    print(f"  Verified {checked} weight tensors")
    print("PASSED: weight_copying")


# ---------------------------------------------------------------------------
# Test 5: add_sub_supernet (sparse aggregation)
# ---------------------------------------------------------------------------
def test_add_sub_supernet(server, sub_supernet, sub_info):
    print("\n=== Test 5: add_sub_supernet ===")

    if sub_info['is_full_supernet']:
        print("SKIPPED: full supernet")
        return

    global_state = server.model.state_dict()
    shared_sum   = {k: torch.zeros_like(v) for k, v in global_state.items()}
    shared_count = {k: torch.zeros_like(v) for k, v in global_state.items()}

    sub_state = sub_supernet.state_dict()
    for key in sub_state:
        if 'num_batches_tracked' not in key:
            sub_state[key] = sub_state[key] + torch.randn_like(sub_state[key]) * 0.01

    mapping = sub_info['mapping']
    server.add_sub_supernet(shared_sum, shared_count, sub_state, mapping, weight=1.0)

    for sub_key, (global_key, slice_info) in list(mapping.items())[:3]:
        if slice_info.get('type') == 'conv':
            out_ch, in_ch = slice_info['out_ch'], slice_info['in_ch']
            active_count   = shared_count[global_key][:out_ch, :in_ch, :, :]
            inactive_count = shared_count[global_key][out_ch:, :, :, :]
            assert active_count.sum() > 0, "Active region should have count > 0"
            if inactive_count.numel() > 0:
                assert inactive_count.sum() == 0, "Inactive region should be zero"
            print(f"  OK: {global_key} active=[:{out_ch}, :{in_ch}]")

    print("PASSED: add_sub_supernet")


# ---------------------------------------------------------------------------
# Test 6: Forward pass through sub-supernet with cache configs
# ---------------------------------------------------------------------------
def test_sub_supernet_forward(server, sub_supernet, sub_info):
    print("\n=== Test 6: sub_supernet_forward ===")

    expansion_choices = server.arch_params['expansion_ratio_choices']
    sub_cache = sub_info['sub_cache']

    if not sub_cache:
        print("SKIPPED: empty sub_cache")
        return

    x = torch.randn(2, 3, 32, 32)

    for label, arch in [('min', sub_cache[0]), ('max', sub_cache[-1])]:
        # arch['e'] is now stage-level: 4 floats, one per stage
        e_indices = [expansion_choices.index(e) for e in arch['e']]
        sub_supernet.set_active_subnet(
            d=arch['d'],
            e_indices=e_indices,
            w_indices=arch['w_indices']
        )
        y = sub_supernet(x)
        assert y.shape == (2, 100), f"Bad output shape [{label}]: {y.shape}"
        print(f"  [{label}] d={arch['d']} -> out {tuple(y.shape)} OK")

    print("PASSED: sub_supernet_forward")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Sub-Supernet Unit Tests")
    print("=" * 60)

    # Test 1: No server needed
    test_per_position_max_d_network()

    # Remaining tests use a server backed by the synthetic temp-CSV cache
    server = build_server()
    test_compute_sub_supernet_bounds(server)
    test_create_sub_supernet(server)
    sub_supernet, sub_info = _build_low_budget_sub_supernet(server)
    test_weight_copying(server, sub_supernet, sub_info)
    test_add_sub_supernet(server, sub_supernet, sub_info)
    test_sub_supernet_forward(server, sub_supernet, sub_info)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
