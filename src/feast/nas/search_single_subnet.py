"""
Search for a Single Subnet for Deployment

This script runs a single GA search to find the best subnet architecture
that fits within a given MACs (or parameter) budget, using the same
entropy-maximizing fitness function as generate_subnet_cache.py.

Usage:
    python search_single_subnet.py \
        --arch_config_path configs/supernets/4-stage-supernet-cifar100-v2.json \
        --target_macs 500e6 \
        --rho0_constraint 0.51
"""

import numpy as np
import os
import sys
import json
import argparse

try:
    from feast.nas.feast_fitness_maximizer import (
        run_entropy_max_ga,
        calculate_L_and_avg_log_w,
        calculate_effectiveness_rho,
        calculate_entropy_objective,
    )
    from feast.utils.subnet_cost import subnet_macs
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your PYTHONPATH.")
    print(f"Import Error: {e}")
    sys.exit(1)


def main(args):
    print("=" * 72)
    print("  Single Subnet Search for Deployment")
    print("=" * 72)

    # --- Load Architecture Configuration ---
    print(f"Loading architecture config from: {args.arch_config_path}")
    with open(args.arch_config_path, 'r') as f:
        arch_config_params = json.load(f)
        arch_config_params['original_stage_base_channels'] = np.array(
            arch_config_params['original_stage_base_channels']
        )
        arch_config_params['alpha_weights'] = np.array(
            arch_config_params.get('alpha_weights', [1.0] * arch_config_params['num_stages'])
        )

    width_choices = arch_config_params['width_multiplier_choices']
    exp_choices = np.array(arch_config_params['expansion_ratio_choices'])
    depth_choices = np.array(list(range(arch_config_params['max_extra_blocks_per_stage'] + 1)))

    # --- Print search space info ---
    num_stages = arch_config_params['num_stages']
    num_w_indices = num_stages + 1

    min_arch = {
        'd': [0] * num_stages,
        'e': [min(exp_choices)] * num_stages,  # stage-level: one per stage
        'w_indices': [0] * num_w_indices,
    }
    max_arch = {
        'd': [max(depth_choices)] * num_stages,
        'e': [max(exp_choices)] * num_stages,  # stage-level: one per stage
        'w_indices': [len(width_choices) - 1] * num_w_indices,
    }
    macs_min, params_min = subnet_macs(min_arch['d'], min_arch['e'], min_arch['w_indices'],
                                        width_mult_options=width_choices, arch_config_params=arch_config_params)
    macs_max, params_max = subnet_macs(max_arch['d'], max_arch['e'], max_arch['w_indices'],
                                        width_mult_options=width_choices, arch_config_params=arch_config_params)

    print(f"\nSearch Space:")
    print(f"  Min subnet: {macs_min/1e6:.2f}M MACs, {params_min/1e6:.2f}M params")
    print(f"  Max subnet: {macs_max/1e6:.2f}M MACs, {params_max/1e6:.2f}M params")
    print(f"\nTarget: {args.target_macs/1e6:.2f}M MACs")
    print(f"Rho0 constraint: {args.rho0_constraint}")
    print(f"GA Settings: pop_size={args.ga_pop_size}, generations={args.ga_generations}, mutate_p={args.ga_mutate_p}")
    print(f"Seed: {args.seed}")
    print("-" * 72)

    # --- Validate target ---
    if args.target_macs < macs_min:
        print(f"\nWARNING: Target MACs ({args.target_macs/1e6:.2f}M) is below the minimum "
              f"possible subnet ({macs_min/1e6:.2f}M). The GA may not find a feasible solution.")
    if args.target_macs > macs_max:
        print(f"\nWARNING: Target MACs ({args.target_macs/1e6:.2f}M) exceeds the maximum "
              f"possible subnet ({macs_max/1e6:.2f}M). The search will return the max subnet.")

    # --- Run GA Search ---
    best_arch, fitness = run_entropy_max_ga(
        mac_budget=args.target_macs,
        arch_config_params=arch_config_params,
        width_mult_options=width_choices,
        depth_choices=depth_choices,
        exp_opt_values=exp_choices,
        rho0_constraint=args.rho0_constraint,
        pop_size=args.ga_pop_size,
        generations=args.ga_generations,
        mutate_p=args.ga_mutate_p,
        seed=args.seed,
        # Optional latency constraints
        lpm_model_combined_path=args.lpm_model_path,
        latency_budget_ms=args.latency_budget_ms,
        latency_fitness_weight=args.latency_fitness_weight,
        lpm_device=args.lpm_device,
        num_processes=args.num_processes,
    )

    # --- Display Results ---
    print("\n" + "=" * 72)
    if best_arch is None:
        print("  SEARCH FAILED: No feasible subnet found for the given constraints.")
        print("  Try relaxing the MACs target or rho0 constraint.")
        print("=" * 72)
        sys.exit(1)

    final_macs, final_params = subnet_macs(
        best_arch['d'], best_arch['e'], best_arch['w_indices'],
        width_mult_options=width_choices, arch_config_params=arch_config_params
    )
    L, avg_log_w = calculate_L_and_avg_log_w(
        best_arch['d'], best_arch['e'], best_arch['w_indices'],
        width_mult_options=width_choices, arch_config_params=arch_config_params
    )
    rho = calculate_effectiveness_rho(L, avg_log_w)
    entropy, _ = calculate_entropy_objective(
        best_arch['d'], best_arch['e'], best_arch['w_indices'],
        width_mult_options=width_choices, arch_config_params=arch_config_params
    )

    print("  SEARCH RESULT — Best Subnet Found")
    print("=" * 72)
    print(f"  MACs:          {final_macs/1e6:.2f}M")
    print(f"  Parameters:    {final_params/1e6:.2f}M")
    print(f"  Fitness Score: {fitness:.6f}")
    print(f"  Effectiveness: {rho:.6f}")
    print(f"  Entropy:       {entropy:.6f}")
    print(f"")
    print(f"  Architecture:")
    print(f"    d (depths):     {best_arch['d']}")
    print(f"    e (expansions): {[float(x) for x in best_arch['e']]}")
    print(f"    w_indices:      {best_arch['w_indices']}")
    print("=" * 72)

    # --- Save result as JSON ---
    if args.output_json:
        result = {
            'target_macs': args.target_macs,
            'macs': final_macs,
            'params': final_params,
            'fitness_score': fitness,
            'effectiveness': rho,
            'entropy': entropy,
            'd': best_arch['d'],
            'e': [float(x) for x in best_arch['e']],
            'w_indices': best_arch['w_indices'],
            'ga_settings': {
                'pop_size': args.ga_pop_size,
                'generations': args.ga_generations,
                'mutate_p': args.ga_mutate_p,
                'rho0_constraint': args.rho0_constraint,
                'seed': args.seed,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for a single optimal subnet for deployment."
    )
    # Required
    parser.add_argument('--arch_config_path', type=str, required=True,
                        help='Path to the JSON file describing the supernet architecture.')
    parser.add_argument('--target_macs', type=float, required=True,
                        help='Target MACs budget for the subnet (e.g., 500e6 for 500M MACs).')

    # GA Settings
    parser.add_argument('--rho0_constraint', type=float, default=0.51,
                        help='Effectiveness (rho) constraint for the GA.')
    parser.add_argument('--ga_pop_size', type=int, default=256,
                        help='Population size for the genetic algorithm.')
    parser.add_argument('--ga_generations', type=int, default=512,
                        help='Number of generations for the genetic algorithm.')
    parser.add_argument('--ga_mutate_p', type=float, default=0.3,
                        help='Mutation probability for the genetic algorithm.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    # Optional Latency Constraints
    parser.add_argument('--lpm_model_path', type=str, default=None,
                        help='Path to the .pth file for latency predictor model (optional).')
    parser.add_argument('--latency_budget_ms', type=float, default=None,
                        help='Latency budget in ms (optional, requires --lpm_model_path).')
    parser.add_argument('--latency_fitness_weight', type=float, default=0.0,
                        help='Weight for latency soft objective in fitness (default: 0.0).')
    parser.add_argument('--lpm_device', type=str, default='cpu',
                        help='Device for latency predictor inference (default: cpu).')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of CPU processes to use (default: all available).')

    # Output
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save the result as a JSON file (optional).')

    args = parser.parse_args()
    main(args)
