"""
Shared utilities for all baseline trainers.

Budget generation is an exact replica of train.py:_precompute_client_budgets()
so that all baselines receive the same per-client MAC budgets as FEAST.
"""

import numpy as np


def get_client_budgets(
    num_clients: int,
    max_mac: float,
    min_mac: float = 25_000_000,
    zipf_alpha: float = 1.2,
    seed: int = 0,
) -> dict:
    """
    Assign per-client MAC budgets using the exact same logic as
    train.py:_precompute_client_budgets() (the FEAST experiment).

    Algorithm (mirrors generic_server_model.py assign_client_resources):
      1. Compute Zipf probabilities over ranks 1..N
      2. Sample N level-indices WITH REPLACEMENT from those probabilities
      3. Map each level-index linearly onto [min_mac, max_mac]
      4. Add uniform jitter ±(step/2), clip to [min_mac, max_mac]
      5. Shuffle so client IDs are not ordered by budget

    Uses numpy global random state (np.random.seed) to match FEAST's seeding,
    which calls np.random.seed(init_seed) before budget generation.

    Returns:
        dict mapping client_idx → budget_macs (float)
    """
    np.random.seed(seed)

    ranks = np.arange(1, num_clients + 1)
    probs = 1.0 / np.power(ranks, zipf_alpha)
    probs /= probs.sum()

    level_indices = np.random.choice(ranks, size=num_clients, p=probs)

    steps = (max_mac - min_mac) / max(1, num_clients - 1)
    budgets = min_mac + (level_indices - 1) * steps

    jitter = np.random.uniform(-steps / 2, steps / 2, size=num_clients)
    budgets = np.clip(budgets + jitter, min_mac, max_mac)

    np.random.shuffle(budgets)

    return {i: float(budgets[i]) for i in range(num_clients)}
