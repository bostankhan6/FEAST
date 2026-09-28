"""
Pytest coverage for the sub-supernet-under-Zipf-budgets mechanism.

The full validation (real cache, prints the paper's communication-reduction
number) lives in scripts/reproduce_sub_supernet_communication_reduction.py --
run that directly to reproduce the reported number. This file exercises the
same correctness/headroom/forward-pass checks against a fast synthetic cache
so they run in CI without depending on the real GA-searched cache CSV.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from reproduce_sub_supernet_communication_reduction import (
    build_server,
    validate_all_configs,
    analyze_headroom,
    spot_check_forward,
)

NUM_CLIENTS = 20


def build_synthetic_server():
    return build_server(use_real_cache=False, num_clients=NUM_CLIENTS,
                         zipf_alpha=1.2, force_max=2, init_seed=0)


def test_all_configs_activate_under_zipf_budgets():
    server = build_synthetic_server()
    failures, unique_budgets = validate_all_configs(server, NUM_CLIENTS)
    assert not failures, f"set_active_subnet failures: {failures}"
    assert len(unique_budgets) > 1, "Zipf sampling should produce more than one budget tier"


def test_sub_supernet_bounds_have_no_headroom():
    server = build_synthetic_server()
    _, unique_budgets = validate_all_configs(server, NUM_CLIENTS)
    assert analyze_headroom(server, unique_budgets)


def test_sub_supernet_forward_pass_across_tiers():
    server = build_synthetic_server()
    _, unique_budgets = validate_all_configs(server, NUM_CLIENTS)
    assert spot_check_forward(server, unique_budgets)
