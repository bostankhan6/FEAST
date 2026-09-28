"""
Builds a fixed variant cache by running run_entropy_max_ga (Supplementary Eq.
offline_cache_search, Sec. B) once per target MACs, spanning a range of budgets
rather than a single one (see search_single_subnet.py for the single-target case).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import json
import argparse
import time

try:
    from feast.nas.feast_fitness_maximizer import (
        run_entropy_max_ga,
        calculate_fitness,
        decode_chromosome,
        calculate_L_and_avg_log_w,
        calculate_effectiveness_rho,
        calculate_entropy_objective
    )
    from feast.utils.subnet_cost import subnet_macs
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your PYTHONPATH.")
    print(f"Import Error: {e}")
    sys.exit(1)

# ==========================================================================================
# --- Job Function (runs serially within the main process) ---
# ==========================================================================================

def run_single_ga_job(job_args):
    """Runs one GA search for a specific MACs target."""
    # Unpack all arguments passed from the main process
    target_macs, job_seed, arch_config_params, width_choices, exp_choices, depth_choices, rho_constraint, ga_pop_size, ga_generations, ga_mutate_p = job_args
    
    print(f"  --> Starting GA for target MACs: {target_macs/1e6:.2f}M (Seed: {job_seed})")
    start_time_job = time.time()

    best_arch, fitness = run_entropy_max_ga(
        mac_budget=target_macs,
        arch_config_params=arch_config_params,
        width_mult_options=width_choices,
        depth_choices=depth_choices,
        exp_opt_values=exp_choices,
        rho0_constraint=rho_constraint,
        pop_size=ga_pop_size,
        generations=ga_generations,
        mutate_p=ga_mutate_p,
        seed=job_seed
    )

    if best_arch:
        final_macs, final_params = subnet_macs(
            best_arch['d'], best_arch['e'], best_arch['w_indices'],
            width_mult_options=width_choices,
            arch_config_params=arch_config_params
        )
        
        # --- Calculate effectiveness and entropy ---
        L, avg_log_w = calculate_L_and_avg_log_w(
            best_arch['d'], best_arch['e'], best_arch['w_indices'],
            width_mult_options=width_choices,
            arch_config_params=arch_config_params
        )
        rho = calculate_effectiveness_rho(L, avg_log_w)

        entropy, _ = calculate_entropy_objective(
            best_arch['d'], best_arch['e'], best_arch['w_indices'],
            width_mult_options=width_choices,
            arch_config_params=arch_config_params
        )
        
        end_time_job = time.time()
        duration_job = end_time_job - start_time_job
        print(f"  <-- Finished GA for target MACs: {target_macs/1e6:.2f}M (Duration: {duration_job:.2f}s)")

        # Return a dictionary with all results in the desired format
        return {
            'macs': final_macs,
            'params': final_params,
            'fitness_score': fitness,
            'effectiveness': rho,
            'entropy': entropy,
            'd': str(best_arch['d']),
            'e': str([float(x) for x in best_arch['e']]),
            'w_indices': str(best_arch['w_indices']),
            'job_target_macs': target_macs, # Add target MACs for debugging/tracking
            'job_seed': job_seed, # Add job seed
            'job_duration_s': duration_job
        }
    
    end_time_job = time.time()
    duration_job = end_time_job - start_time_job
    print(f"  <-- GA for target MACs: {target_macs/1e6:.2f}M FAILED to find best architecture (Duration: {duration_job:.2f}s)")
    return None # Return None if no best_arch was found

# ==========================================================================================
# --- Main Script Logic ---
# ==========================================================================================

def main(args):
    print("--- Flexible Fitness Dataset Generation Script (Serial Jobs) ---")
    
    # --- Load Architecture Configuration from JSON file ---
    print(f"Loading architecture config from: {args.arch_config_path}")
    with open(args.arch_config_path, 'r') as f:
        arch_config_params = json.load(f)
        # Convert lists back to numpy arrays where needed for calculations
        arch_config_params['original_stage_base_channels'] = np.array(arch_config_params['original_stage_base_channels'])
        arch_config_params['alpha_weights'] = np.array(arch_config_params.get('alpha_weights', [1.0] * arch_config_params['num_stages']))

    # Define search space choices from the loaded config
    width_multiplier_choices = arch_config_params['width_multiplier_choices']
    expansion_ratio_choices = np.array(arch_config_params['expansion_ratio_choices'])
    depth_choices = np.array(list(range(arch_config_params['max_extra_blocks_per_stage'] + 1)))

    collected_data = []
    if os.path.exists(args.output_csv):
        print(f"Resuming from existing dataset: '{args.output_csv}'")
        df_existing = pd.read_csv(args.output_csv)
        collected_data = df_existing.to_dict('records')
    
    num_samples_needed = args.num_samples - len(collected_data)
    if num_samples_needed <= 0:
        print(f"Dataset already has {len(collected_data)} samples. No new samples needed.")
        return

    print(f"Targeting {args.num_samples} total samples. Need to generate {num_samples_needed} more.")

    # --- Determine MACs bounds based on selected mode ---
    if args.bounds_mode == 'absolute':
        print("Using absolute MACs bounds of the supernet.")
        num_stages = arch_config_params['num_stages']
        num_w_indices = num_stages + 1
        min_arch_config = {'d': [0]*num_stages, 'e': [min(expansion_ratio_choices)]*num_stages, 'w_indices': [0]*num_w_indices}
        max_arch_config = {'d': [max(depth_choices)]*num_stages, 'e': [max(expansion_ratio_choices)]*num_stages, 'w_indices': [len(width_multiplier_choices)-1]*num_w_indices}
        
        macs_min, _ = subnet_macs(min_arch_config['d'], min_arch_config['e'], min_arch_config['w_indices'], width_mult_options=width_multiplier_choices, arch_config_params=arch_config_params)
        macs_max, _ = subnet_macs(max_arch_config['d'], max_arch_config['e'], max_arch_config['w_indices'], width_mult_options=width_multiplier_choices, arch_config_params=arch_config_params)
    elif args.bounds_mode == 'relative':
        print("Using user-defined relative MACs bounds.")
        macs_min, macs_max = args.macs_lower_bound, args.macs_upper_bound
    else:
        raise ValueError("Invalid --bounds_mode. Choose 'absolute' or 'relative'.")
        
    print(f"Search will operate in MACs range: [{macs_min/1e6:.2f}M, {macs_max/1e6:.2f}M]")

    # --- Generate MACs targets based on selected mode ---
    if args.sampling_mode == 'random':
        print(f"Using random sampling for {num_samples_needed} MACs targets.")
        # Ensure consistent random seeds for the job generation
        np.random.seed(args.base_seed)
        mac_targets = np.random.uniform(macs_min, macs_max, num_samples_needed)
    elif args.sampling_mode == 'equidistant':
        print(f"Using equidistant sampling for {num_samples_needed} MACs targets.")
        mac_targets = np.linspace(macs_min, macs_max, num_samples_needed)
    else:
        raise ValueError("Invalid --sampling_mode. Choose 'random' or 'equidistant'.")
        
    # Prepare arguments for serial jobs
    # Each job will get a unique seed derived from the base_seed + its index
    job_args_list = [(mac, args.base_seed + i, arch_config_params, width_multiplier_choices, expansion_ratio_choices, depth_choices, args.rho0_constraint, args.ga_pop_size, args.ga_generations, args.ga_mutate_p) for i, mac in enumerate(mac_targets)]

    # --- Run serial generation ---
    print(f"Starting serial generation of {num_samples_needed} samples...")
    pbar = tqdm(total=num_samples_needed, desc="Generating Samples Serially")
    
    current_sample_count = len(collected_data) # Keep track for saving

    for i, job_arg in enumerate(job_args_list):
        result = run_single_ga_job(job_arg)
        if result:
            collected_data.append(result)
            current_sample_count += 1 # Increment only for successful samples
        pbar.update(1) # Always update pbar, even if a job fails
        
        # Save progress at intervals
        if (i + 1) % args.save_interval == 0 and collected_data:
            df = pd.DataFrame(collected_data)
            df.to_csv(args.output_csv, index=False)
            tqdm.write(f"  -> Saved {len(collected_data)} samples to {args.output_csv} (after {i+1} jobs processed)")
    
    pbar.close()

    # --- Final save ---
    if collected_data:
        df = pd.DataFrame(collected_data)
        # Sort and reorder columns
        df = df.sort_values(by='macs').reset_index(drop=True)
        cols = ['macs', 'params', 'fitness_score', 'effectiveness', 'entropy', 'd', 'e', 'w_indices', 'job_target_macs', 'job_seed', 'job_duration_s']
        # Filter columns to only include those present in df
        df = df[[col for col in cols if col in df.columns]] 
        df.to_csv(args.output_csv, index=False)
        print(f"\nFinished generation. Final dataset with {len(collected_data)} samples saved to {args.output_csv}.")
    else:
        print("\nNo samples were successfully generated and collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible Fitness Dataset Generation Script")
    parser.add_argument('--arch_config_path', type=str, required=True, help='Path to the JSON file describing the supernet architecture.')
    parser.add_argument('--output_csv', type=str, default="generated_cache.csv", help='Path to save the output CSV file.')
    
    # Search Strategy Arguments
    parser.add_argument('--bounds_mode', type=str, default='absolute', choices=['absolute', 'relative'], help='Set MACs bounds to absolute supernet extremes or a relative range.')
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['random', 'equidistant'], help='How to sample MACs targets within the bounds.')
    parser.add_argument('--macs_lower_bound', type=float, default=400e6, help='Lower MACs bound for relative mode.')
    parser.add_argument('--macs_upper_bound', type=float, default=3400e6, help='Upper MACs bound for relative mode.')
    
    # Generation and GA Arguments
    parser.add_argument('--num_samples', type=int, default=60, help='Total number of subnet samples to generate for the cache.')
    parser.add_argument('--rho0_constraint', type=float, default=0.31, help='Effectiveness (rho) constraint for the GA.')
    parser.add_argument('--ga_pop_size', type=int, default=128, help='Population size for the genetic algorithm.')
    parser.add_argument('--ga_generations', type=int, default=128, help='Number of generations for the genetic algorithm.')
    parser.add_argument('--ga_mutate_p', type=float, default=0.3, help='Mutation probability for the genetic algorithm.')
    
    # System Arguments
    parser.add_argument('--save_interval', type=int, default=10, help='Save progress every N samples.')
    parser.add_argument('--base_seed', type=int, default=42, help='Base seed for random number generation for MAC targets and job-specific seeds.')
    
    args = parser.parse_args()
    main(args)