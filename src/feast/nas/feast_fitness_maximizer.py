import numpy as np
from tqdm import tqdm
import os
import sys
import multiprocessing
from functools import partial
from typing import Optional

# For Latency Predictor MLP
import torch
import torch.nn as nn

try:
    from feast.utils.subnet_cost import subnet_macs
except ImportError:
    print("ERROR: Could not import 'subnet_macs' from subnet_macs.py.")
    sys.exit(1)

# ==========================================================================================
# --- CONSTANTS ---
# ==========================================================================================
K_MAIN_CONV = 3.0
K_PROJ_CONV = 1.0
GROUPS = 1.0
EPS = 1e-9

# ==========================================================================================
# --- UTILITY FUNCTIONS (Common to both scripts or specific to GA) ---
# ==========================================================================================

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return int(new_v)

def _calculate_dynamic_stage_output_hws(arch_config_params):
    initial_hw = arch_config_params['initial_input_hw']
    stem_stride = arch_config_params['stem_stride']
    num_stages = arch_config_params['num_stages']
    stage_downsample_factors = arch_config_params['stage_downsample_factors']
    output_hws = []
    current_hw = initial_hw // stem_stride
    for i in range(num_stages):
        current_hw = current_hw // stage_downsample_factors[i]
        output_hws.append(current_hw)
    return np.array(output_hws)

def calculate_H_j_for_stage(
    stage_idx, d_extra_block, e_ratio_for_stage,
    scaled_c_in_to_stage, scaled_c_out_of_stage,
    stage_output_hw_for_entropy, arch_config_params):
    channel_divisible_by = arch_config_params['channel_divisible_by']
    log_output_volume_term = np.log(stage_output_hw_for_entropy**2 * scaled_c_out_of_stage + EPS)
    sum_log_widths_term = 0.0
    current_scaled_c_in_to_block = float(scaled_c_in_to_stage)
    num_blocks_in_stage = 1 + d_extra_block
    for block_num in range(num_blocks_in_stage):
        current_exp_ratio = e_ratio_for_stage  # same ratio for all blocks in this stage
        scaled_c_mid_conv1 = float(make_divisible(scaled_c_out_of_stage * current_exp_ratio, divisor=channel_divisible_by))
        if scaled_c_mid_conv1 < channel_divisible_by: scaled_c_mid_conv1 = float(channel_divisible_by)
        width_conv1 = current_scaled_c_in_to_block * (K_MAIN_CONV**2) / GROUPS
        sum_log_widths_term += np.log(width_conv1 + EPS if width_conv1 > 0 else EPS)
        width_conv2 = scaled_c_mid_conv1 * (K_MAIN_CONV**2) / GROUPS
        sum_log_widths_term += np.log(width_conv2 + EPS if width_conv2 > 0 else EPS)
        if block_num == 0:
            width_proj = float(scaled_c_in_to_stage) * (K_PROJ_CONV**2) / GROUPS
            sum_log_widths_term += np.log(width_proj + EPS if width_proj > 0 else EPS)
        current_scaled_c_in_to_block = scaled_c_out_of_stage
    if np.isneginf(log_output_volume_term) or np.isneginf(sum_log_widths_term) or \
       np.isnan(log_output_volume_term) or np.isnan(sum_log_widths_term):
        return 0.0
    return log_output_volume_term * sum_log_widths_term

def calculate_entropy_objective(depth_vec, exp_vec_values, w_indices, width_mult_options, arch_config_params):
    H_j_list = []
    num_stages = arch_config_params['num_stages']
    original_stem_out_channels = arch_config_params['original_stem_out_channels']
    original_stage_base_channels = arch_config_params['original_stage_base_channels']
    stage_output_hws_for_entropy = _calculate_dynamic_stage_output_hws(arch_config_params)
    alpha_weights = np.array(arch_config_params.get('alpha_weights', [1.0]*num_stages))
    channel_divisible_by = arch_config_params['channel_divisible_by']
    stem_width_multiplier = width_mult_options[w_indices[0]]
    scaled_stem_cout = make_divisible(original_stem_out_channels * stem_width_multiplier, divisor=channel_divisible_by)
    current_scaled_c_in_to_stage = float(scaled_stem_cout)
    for j in range(num_stages):
        stage_width_multiplier = width_mult_options[w_indices[j+1]]
        scaled_c_out_of_stage = make_divisible(original_stage_base_channels[j] * stage_width_multiplier, divisor=channel_divisible_by)
        if scaled_c_out_of_stage < channel_divisible_by: scaled_c_out_of_stage = channel_divisible_by
        H_j = calculate_H_j_for_stage(
            stage_idx=j, d_extra_block=depth_vec[j], e_ratio_for_stage=exp_vec_values[j],
            scaled_c_in_to_stage=current_scaled_c_in_to_stage, scaled_c_out_of_stage=scaled_c_out_of_stage,
            stage_output_hw_for_entropy=stage_output_hws_for_entropy[j], arch_config_params=arch_config_params
        )
        H_j_list.append(H_j)
        current_scaled_c_in_to_stage = scaled_c_out_of_stage
    weighted_H_sum = np.dot(alpha_weights[:num_stages], np.array(H_j_list))
    return weighted_H_sum, H_j_list

def calculate_L_and_avg_log_w(depth_vec, exp_vec_values, w_indices, width_mult_options, arch_config_params):
    L_total, sum_log_w = 0, 0.0
    num_stages = arch_config_params['num_stages']
    original_stem_out_channels = arch_config_params['original_stem_out_channels']
    original_stage_base_channels = arch_config_params['original_stage_base_channels']
    channel_divisible_by = arch_config_params['channel_divisible_by']
    stem_width_multiplier = width_mult_options[w_indices[0]]
    scaled_stem_cout = make_divisible(original_stem_out_channels * stem_width_multiplier, divisor=channel_divisible_by)
    c_in_stem = arch_config_params.get('initial_input_channels', 3.0)
    w_stem = c_in_stem * (K_MAIN_CONV**2) / GROUPS
    sum_log_w += np.log(w_stem + EPS)
    L_total += 1
    current_scaled_c_in_to_next_stage = float(scaled_stem_cout)
    for stage_idx in range(num_stages):
        stage_width_multiplier = width_mult_options[w_indices[stage_idx+1]]
        scaled_c_out_of_this_stage = make_divisible(original_stage_base_channels[stage_idx] * stage_width_multiplier, divisor=channel_divisible_by)
        if scaled_c_out_of_this_stage < channel_divisible_by: scaled_c_out_of_this_stage = channel_divisible_by
        scaled_c_in_to_first_block_of_stage = current_scaled_c_in_to_next_stage
        num_blocks_in_stage_active = 1 + depth_vec[stage_idx]
        current_exp_ratio = exp_vec_values[stage_idx]  # same ratio for all blocks in this stage
        for block_num in range(num_blocks_in_stage_active):
            current_scaled_c_in_to_block = scaled_c_in_to_first_block_of_stage if block_num == 0 else scaled_c_out_of_this_stage
            scaled_c_mid = float(make_divisible(scaled_c_out_of_this_stage * current_exp_ratio, divisor=channel_divisible_by))
            if scaled_c_mid < channel_divisible_by: scaled_c_mid = float(channel_divisible_by)
            w1 = current_scaled_c_in_to_block * (K_MAIN_CONV**2) / GROUPS
            sum_log_w += np.log(w1 + EPS if w1 > 0 else EPS)
            L_total += 1
            w2 = scaled_c_mid * (K_MAIN_CONV**2) / GROUPS
            sum_log_w += np.log(w2 + EPS if w2 > 0 else EPS)
            L_total += 1
            if block_num == 0:
                w_proj = scaled_c_in_to_first_block_of_stage * (K_PROJ_CONV**2) / GROUPS
                sum_log_w += np.log(w_proj + EPS if w_proj > 0 else EPS)
                L_total += 1
        current_scaled_c_in_to_next_stage = scaled_c_out_of_this_stage
    if L_total == 0: return 0, 0.0
    return L_total, sum_log_w / L_total

def calculate_effectiveness_rho(L_total, avg_log_w):
    if L_total == 0: return float('inf')
    bar_w = np.exp(avg_log_w)
    if bar_w < EPS: return float('inf')
    return L_total / bar_w

def decode_chromosome(chrom, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list):
    depth_genes_end = num_depth_genes
    exp_genes_end = num_depth_genes + num_exp_genes
    d_vec = chrom[:depth_genes_end].astype(int).tolist()
    e_indices = chrom[depth_genes_end:exp_genes_end].astype(int)
    w_indices = chrom[exp_genes_end:exp_genes_end + num_width_idx_genes].astype(int).tolist()
    e_indices = np.clip(e_indices, 0, len(exp_opt_values_list) - 1)
    e_vec_values = [exp_opt_values_list[i] for i in e_indices]
    return {"d": d_vec, "e": e_vec_values, "w_indices": w_indices}

# ==========================================================================================
# --- LATENCY PREDICTOR MLP ---
# ==========================================================================================
class LatencyPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(LatencyPredictorMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1)) # Output is a single latency value
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==========================================================================================
# --- ARCHITECTURE FEATURE EXTRACTION ---
# ==========================================================================================
def arch_to_feature_vector_and_names(arch, supernet_arch_params):
    features = []
    feature_names = []

    # Depth genes (arch["d"] is a list of depths for each stage)
    for i, d in enumerate(arch["d"]):
        features.append(d)
        feature_names.append(f"depth_stage{i}")

    # Expansion ratio genes (arch["e"] is a list of actual expansion ratio values)
    for i, e_ratio in enumerate(arch["e"]):
        features.append(e_ratio)
        feature_names.append(f"exp_ratio_block{i}") 

    # Width multiplier indices (arch["w_indices"] is a list of width multiplier indices)
    width_mult_options = supernet_arch_params['width_multiplier_choices']
    for i, w_idx in enumerate(arch["w_indices"]):
        features.append(width_mult_options[w_idx]) 
        feature_names.append(f"width_mult_stage{i}") 

    # Add MACs and Parameters as features (they are strong indicators for latency)
    macs, params = subnet_macs(
        arch["d"], arch["e"], arch["w_indices"], 
        width_mult_options, supernet_arch_params
    )
    features.append(macs)
    feature_names.append("macs")
    features.append(params)
    feature_names.append("params")

    return np.array(features, dtype=np.float32), feature_names


# ==========================================================================================
# --- LATENCY PREDICTION HELPER FUNCTION ---
# ==========================================================================================
def predict_latency(arch, supernet_arch_params, lpm_model, scaler, device):
    """
    Predicts latency for a given architecture using the trained LPM.
    """
    # 1. Convert architecture dictionary to feature vector
    features, _ = arch_to_feature_vector_and_names(arch, supernet_arch_params)
    
    # 2. Reshape for scaler and apply scaling
    features_reshaped = features.reshape(1, -1) # Scaler expects 2D input
    scaled_features = scaler.transform(features_reshaped)
    
    # 3. Predict latency using LPM
    lpm_model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        # Convert scaled features to tensor and move to device
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        predicted_latency_raw = lpm_model(input_tensor).squeeze().cpu().numpy()
        predicted_latency = max(0.0, predicted_latency_raw.item())
    
    # The result is a numpy array, take the single value
    return predicted_latency # Return as a scalar float


# ==========================================================================================
# --- GLOBAL LPM OBJECTS FOR MULTIPROCESSING WORKERS ---
# ==========================================================================================
# These will be set by the initializer in each worker process
_lpm_model_global = None
_scaler_global = None
_lpm_device_global = None
_lpm_path_global = None

def worker_init_lpm(lpm_model_combined_path, lpm_device):
    """
    Initializer function for multiprocessing pool.
    Loads the LPM model and scaler once per worker process.
    """
    global _lpm_model_global, _scaler_global, _lpm_device_global, _lpm_path_global
    
    if lpm_model_combined_path is None:
        _lpm_model_global = None
        _scaler_global = None
        _lpm_device_global = None
        _lpm_path_global = None
        return

    if not os.path.exists(lpm_model_combined_path):
        print(f"Warning (worker {os.getpid()}): LPM model not found at {lpm_model_combined_path}. Latency prediction will be skipped for this worker.")
        _lpm_model_global = None
        _scaler_global = None
        _lpm_device_global = None
        _lpm_path_global = None
        return

    print(f"Loading LPM in worker {os.getpid()} from: {lpm_model_combined_path} for device {lpm_device}")
    try:
        # Crucially, map_location to CPU during load, then move to desired device if it's CUDA
        checkpoint = torch.load(lpm_model_combined_path, map_location="cpu", weights_only=False) 
        
        input_dim = checkpoint['input_dim']
        hidden_layers = checkpoint['hidden_layers']
        
        lpm_model = LatencyPredictorMLP(input_dim, hidden_layers)
        lpm_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to the target device AFTER loading on CPU
        _lpm_device_global = lpm_device
        if _lpm_device_global.startswith('cuda') and torch.cuda.is_available():
            # Set CUDA device for this worker if specified
            device_idx = int(_lpm_device_global.split(':')[-1])
            torch.cuda.set_device(device_idx)
            lpm_model.to(_lpm_device_global)
            print(f"Worker {os.getpid()} initialized LPM on {_lpm_device_global}")
        else:
            lpm_model.to("cpu") # Ensure it's on CPU if not using CUDA
            _lpm_device_global = "cpu"
            print(f"Worker {os.getpid()} initialized LPM on CPU")

        lpm_model.eval() # Set to evaluation mode
        
        _lpm_model_global = lpm_model
        _scaler_global = checkpoint['scaler']
        _lpm_path_global = lpm_model_combined_path # Store for reference

    except Exception as e:
        print(f"Error loading LPM in worker {os.getpid()}: {e}")
        _lpm_model_global = None
        _scaler_global = None
        _lpm_device_global = None
        _lpm_path_global = None


# ==========================================================================================
# --- FITNESS FUNCTION ---
# ==========================================================================================
def calculate_fitness(
    chromosome, mac_limit, param_limit, constraint_type, rho0_constraint, 
    width_mult_options, exp_opt_values_list, num_depth_genes, num_exp_genes, 
    num_width_idx_genes, arch_config_params_ga, effectiveness_weight,
    
    # Latency-related arguments (LPM objects come from globals set by the pool initializer)
    latency_budget_ms: Optional[float],
    latency_fitness_weight: float,
    
    verbose_debug=False
):
    arch = decode_chromosome(chromosome, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list)
    num_stages_fit = arch_config_params_ga['num_stages']
    original_stage_base_channels_fit = arch_config_params_ga['original_stage_base_channels']
    channel_divisible_by_fit = arch_config_params_ga['channel_divisible_by']
    
    # Non-decreasing channels constraint
    scaled_stage_output_channels = []
    for j in range(num_stages_fit):
        stage_width_multiplier = width_mult_options[arch["w_indices"][j+1]]
        scaled_c_out_of_stage = make_divisible(original_stage_base_channels_fit[j] * stage_width_multiplier, divisor=channel_divisible_by_fit)
        if scaled_c_out_of_stage < channel_divisible_by_fit: scaled_c_out_of_stage = channel_divisible_by_fit
        scaled_stage_output_channels.append(scaled_c_out_of_stage)
    
    quantified_channel_violation = 0.0
    if arch_config_params_ga.get('non_decreasing_channels', True):
        for i in range(len(scaled_stage_output_channels) - 1):
            if scaled_stage_output_channels[i] > scaled_stage_output_channels[i+1]:
                quantified_channel_violation += (scaled_stage_output_channels[i] - scaled_stage_output_channels[i+1])
    
    current_macs, current_params = subnet_macs(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)

    # --- Apply Hard Constraints (MACs, Params, Rho, Latency Budget) ---
    if constraint_type == 'macs' or constraint_type == 'both':
        if current_macs > mac_limit:
            if verbose_debug: print(f"Failed MAC: {current_macs/1e6:.2f}M > {mac_limit/1e6:.2f}M")
            return -1e9 # Very low fitness for invalid architectures
    
    if constraint_type == 'params' or constraint_type == 'both':
        if current_params > param_limit:
            if verbose_debug: print(f"Failed Params: {current_params/1e6:.2f}M > {param_limit/1e6:.2f}M")
            return -1e9

    L_total, avg_log_w = calculate_L_and_avg_log_w(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    rho = calculate_effectiveness_rho(L_total, avg_log_w)
    if rho > rho0_constraint:
        if verbose_debug: print(f"Failed Rho: {rho:.4f} > {rho0_constraint:.4f}")
        return -1e8 # Slightly less severe penalty for rho to differentiate from MACs/Params/Latency if debugging

    # --- Latency Prediction and Budget Check (if enabled) ---
    predicted_latency_ms = None
    # Use global LPM objects here
    if _lpm_model_global is not None and _scaler_global is not None:
        predicted_latency_ms = predict_latency(
            arch=arch, 
            supernet_arch_params=arch_config_params_ga, 
            lpm_model=_lpm_model_global, 
            scaler=_scaler_global, 
            device=_lpm_device_global # Use the device set for this worker
        )
        
        if latency_budget_ms is not None:
            if predicted_latency_ms > latency_budget_ms:
                if verbose_debug: print(f"Failed Latency Budget: {predicted_latency_ms:.2f}ms > {latency_budget_ms:.2f}ms")
                return -1e9 # Very low fitness for exceeding latency budget

    # --- Calculate Soft Objectives and Combine Fitness ---
    raw_paper_entropy_score, _ = calculate_entropy_objective(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    actual_stage_depths = [1 + d_extra for d_extra in arch["d"]]
    variance_of_depths = np.var(actual_stage_depths)
    Q_penalty_term = np.exp(variance_of_depths)
    beta_depth_penalty = arch_config_params_ga.get('beta_depth_penalty', 10.0)
    penalized_entropy_score = raw_paper_entropy_score - beta_depth_penalty * Q_penalty_term
    
    # Start with entropy and effectiveness
    combined_fitness = penalized_entropy_score + effectiveness_weight * rho

    # Add non-decreasing channel penalty if applicable
    if arch_config_params_ga.get('non_decreasing_channels', True) and quantified_channel_violation > 0:
        channel_penalty_coeff = arch_config_params_ga.get('non_decreasing_penalty_coeff', 1.0)
        combined_fitness -= channel_penalty_coeff * quantified_channel_violation
        if verbose_debug:
            print(f"    Applied Non-Decreasing Channel Penalty. ViolationSum: {quantified_channel_violation:.2f}, Coeff: {channel_penalty_coeff}, PenaltyValue: {channel_penalty_coeff * quantified_channel_violation:.2f}")

    # Add latency soft objective if enabled and weight is non-zero
    if predicted_latency_ms is not None and latency_fitness_weight != 0.0:
        # Since lower latency is better, we add a negative term to maximize fitness
        combined_fitness += latency_fitness_weight * predicted_latency_ms
        if verbose_debug:
            print(f"    Applied Latency Soft Objective. Predicted Latency: {predicted_latency_ms:.2f}ms, Coeff: {latency_fitness_weight}, PenaltyValue: {latency_fitness_weight * predicted_latency_ms:.2f}")

    if verbose_debug:
        macs_str = f"MACs={current_macs/1e6:.2f}M"
        params_str = f"Params={current_params/1e6:.2f}M"
        rho_str = f"ρ={rho:.4f}"
        latency_str = f"Latency={predicted_latency_ms:.2f}ms" if predicted_latency_ms is not None else ""
        tqdm.write(f"PASSED: {macs_str}, {params_str}, {rho_str}, {latency_str}, CombinedFitness={combined_fitness:.4f}")
    
    return combined_fitness

def _calculate_fitness_worker(fixed_args, chromosome):
    """ Worker function for multiprocessing. Receives chromosome as the first arg. """
    mac_limit, param_limit, constraint_type, rho0_constraint, width_mult_options, exp_opt_values_list, \
    num_depth_genes, num_exp_genes, num_width_idx_genes, \
    arch_config_params_ga, effectiveness_weight, \
    latency_budget_ms, latency_fitness_weight = fixed_args # Removed LPM-specific objects

    return calculate_fitness(chromosome, mac_limit, param_limit, constraint_type, rho0_constraint,
                             width_mult_options, exp_opt_values_list,
                             num_depth_genes, num_exp_genes, num_width_idx_genes,
                             arch_config_params_ga, effectiveness_weight,
                             latency_budget_ms, latency_fitness_weight)


# ==========================================================================================
# ------------ GA EVOLUTIONARY SEARCH -------------------------------
# ==========================================================================================
def run_entropy_max_ga(
    mac_budget: Optional[float] = None,
    param_budget: Optional[float] = None,
    constraint_type: str = 'macs', # 'macs', 'params', 'both'

    arch_config_params: Optional[dict] = None, # Changed to optional for flexibility; ensure it's provided if needed
    width_mult_options: Optional[list] = None, # Changed to optional
    depth_choices: Optional[np.ndarray] = None, # Changed to optional
    exp_opt_values: Optional[list] = None,      # Changed to optional
    rho0_constraint: float = 2.0,
    effectiveness_fitness_weight: float = 1000,
    pop_size: int = 64,
    generations: int = 25,
    mutate_p: float = 0.3,
    seed: int = 0,
    max_init_attempts_factor: int = 2000,

    # Optional latency-related arguments
    lpm_model_combined_path: Optional[str] = None, # Path to the .pth file containing model_state_dict, input_dim, hidden_layers, and scaler
    latency_budget_ms: Optional[float] = None,
    latency_fitness_weight: float = 0.0, # Default to 0.0 for no soft objective by default
    lpm_device: str = "cpu", # Device where LPM will perform inference

    verbose_debug_ga: bool = False, # Enable verbose debugging in the GA loop
    num_processes: Optional[int] = None # Number of worker processes for fitness evaluation
    ):
    """
    Genetic-algorithm search for the entropy/effectiveness-maximizing architecture
    under a MAC or parameter budget -- the offline cache-construction procedure of
    Supplementary Eq. offline_cache_search (Sec. B): argmax F_DFN(a) s.t. MAC(a) <=
    B_nu, rho(a) <= rho0_nu. F_DFN (calculate_entropy_objective) and rho
    (calculate_effectiveness_rho) are the training-free fitness and effectiveness
    measures defined by the cited DeepFedNAS work, reused here unmodified for
    constructing the fixed variant cache at each target budget.
    """
    # --- Input Validation for Constraints ---
    if constraint_type not in ['macs', 'params', 'both']:
        raise ValueError("constraint_type must be 'macs', 'params', or 'both'")
    
    if constraint_type == 'macs' and mac_budget is None:
        raise ValueError("mac_budget must be provided if constraint_type is 'macs'")
    if constraint_type == 'params' and param_budget is None:
        raise ValueError("param_budget must be provided if constraint_type is 'params'")
    if constraint_type == 'both' and (mac_budget is None or param_budget is None):
        raise ValueError("mac_budget and param_budget must be provided if constraint_type is 'both'")
    
    # Optional warnings if extra budgets are provided but not used
    if constraint_type == 'macs' and param_budget is not None:
          print("Warning: param_budget provided but constraint_type is 'macs'. param_budget will be ignored.")
    if constraint_type == 'params' and mac_budget is not None:
          print("Warning: mac_budget provided but constraint_type is 'params'. mac_budget will be ignored.")

    # --- Input Validation for Arch Config and Choices ---
    if arch_config_params is None or width_mult_options is None or depth_choices is None or exp_opt_values is None:
        raise ValueError("arch_config_params, width_mult_options, depth_choices, and exp_opt_values must be provided.")

    num_stages = arch_config_params['num_stages']
    max_extra_blocks_per_stage = arch_config_params['max_extra_blocks_per_stage']
    num_depth_genes_ga = num_stages
    num_exp_genes_ga = num_stages  # one expansion ratio per stage (stage-level)
    num_width_idx_genes_ga = num_stages + 1
    chromosome_len_ga = num_depth_genes_ga + num_exp_genes_ga + num_width_idx_genes_ga

    rng = np.random.default_rng(seed)
    
    print(f"\nStarting GA: Constraint Type: {constraint_type.upper()}")
    if mac_budget is not None:
        print(f"   MAC Budget: {mac_budget/1e6:.2f} M")
    if param_budget is not None:
        print(f"   Parameter Budget: {param_budget/1e6:.2f} M")
    print(f"   ρ₀ <= {rho0_constraint}")

    if lpm_model_combined_path is not None:
        print(f"LPM path provided: {lpm_model_combined_path}. LPM device for GA: {lpm_device}")
        if latency_budget_ms is not None:
            print(f"   Latency Budget: {latency_budget_ms:.2f} ms")
        if latency_fitness_weight != 0.0:
            print(f"   Latency Fitness Weight: {latency_fitness_weight}")
    else:
        print("Latency optimization is NOT enabled.")


    # --- Prepare for Parallelization ---
    # These args are constant for all fitness evaluations
    fixed_args = (
        mac_budget, param_budget, constraint_type, rho0_constraint, width_mult_options, exp_opt_values,
        num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
        arch_config_params, effectiveness_fitness_weight,
        latency_budget_ms, latency_fitness_weight
    )
    worker_func = partial(_calculate_fitness_worker, fixed_args)

    # Use initializer to load LPM once per worker process
    # Pass path and target device to initializer
    pool_initializer = partial(worker_init_lpm, lpm_model_combined_path, lpm_device)

    with multiprocessing.Pool(processes=num_processes, initializer=pool_initializer) as pool:
        # --- Parallel Initialization ---
        pop = []
        pbar_init = tqdm(total=pop_size, desc="Initializing Population", ncols=120)
        init_attempts_count = 0
        init_valid_fitness_threshold = -1e7
        while len(pop) < pop_size and init_attempts_count < max_init_attempts_factor * pop_size:
            batch_size_init = max(1, (pop_size - len(pop)) * 5)
            candidate_chromosomes = []
            for _ in range(batch_size_init):
                init_attempts_count += 1
                depth_g = rng.choice(depth_choices, num_depth_genes_ga)
                exp_g_indices = rng.integers(0, len(exp_opt_values), num_exp_genes_ga)
                w_g_indices = rng.integers(0, len(width_mult_options), num_width_idx_genes_ga)
                candidate_chromosomes.append(np.concatenate([depth_g, exp_g_indices, w_g_indices]))
            
            fitness_scores = pool.map(worker_func, candidate_chromosomes)
            
            valid_chromosomes = [chromo for chromo, score in zip(candidate_chromosomes, fitness_scores) if score > init_valid_fitness_threshold]
            
            needed = pop_size - len(pop)
            pop.extend(valid_chromosomes[:needed])
            pbar_init.update(len(pop) - pbar_init.n)
            pbar_init.set_postfix_str(f"Generated: {init_attempts_count}, Found: {len(pop)}")
        
        pbar_init.close()

        if not pop:
            print(f"ERROR: Could not initialize ANY valid individuals after {init_attempts_count} attempts. Check your constraints or budget.")
            return None, -float('inf')
        
        pop = np.stack(pop)
        best_overall_fitness = -float('inf')
        best_overall_chromosome = None

        # --- GA Generations with Parallel Fitness Evaluation ---
        for gen in tqdm(range(generations), desc="GA Generations", ncols=120):
            current_pop_fitness_scores = np.array(pool.map(worker_func, pop))

            gen_best_fitness_idx = np.argmax(current_pop_fitness_scores)
            if current_pop_fitness_scores[gen_best_fitness_idx] > best_overall_fitness:
                best_overall_fitness = current_pop_fitness_scores[gen_best_fitness_idx]
                best_overall_chromosome = pop[gen_best_fitness_idx].copy()
            
            # Selection (Tournament Selection)
            parents = []
            num_parents_to_select = len(pop)
            for _ in range(num_parents_to_select):
                competitor_indices = rng.choice(len(pop), 3, replace=False)
                winner_idx = competitor_indices[np.argmax(current_pop_fitness_scores[competitor_indices])]
                parents.append(pop[winner_idx])
            parents = np.array(parents)

            # Crossover (Single Point)
            children = []
            for i in range(0, len(parents), 2):
                if i + 1 >= len(parents):
                    children.append(parents[i])
                    continue
                p1, p2 = parents[i], parents[i+1]
                cx_point = rng.integers(1, chromosome_len_ga)
                children.append(np.concatenate((p1[:cx_point], p2[cx_point:])))
                children.append(np.concatenate((p2[:cx_point], p1[cx_point:])))
            
            pop = np.array(children[:len(pop)])

            # Mutation
            for i in range(len(pop)):
                if rng.random() < mutate_p:
                    gene_idx = rng.integers(chromosome_len_ga)
                    if gene_idx < num_depth_genes_ga:
                        pop[i, gene_idx] = rng.choice(depth_choices)
                    elif gene_idx < num_depth_genes_ga + num_exp_genes_ga:
                        pop[i, gene_idx] = rng.integers(len(exp_opt_values))
                    else:
                        pop[i, gene_idx] = rng.integers(0, len(width_mult_options))

            # Elitism: Ensure the best overall chromosome survives
            if best_overall_chromosome is not None and len(pop) > 0:
                # Need to calculate fitness of best_overall_chromosome directly as it's not in current pop
                best_chromo_fitness_for_elitism = calculate_fitness(
                    best_overall_chromosome, mac_budget, param_budget, constraint_type, rho0_constraint, width_mult_options, exp_opt_values,
                    num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga, arch_config_params, effectiveness_fitness_weight,
                    latency_budget_ms, latency_fitness_weight, verbose_debug=False
                )
                
                # We need current population fitness to find the worst child to replace
                child_fitness_scores_for_elitism = np.array(pool.map(worker_func, pop))
                
                if child_fitness_scores_for_elitism.size > 0:
                    worst_child_idx = np.argmin(child_fitness_scores_for_elitism)
                    if best_chromo_fitness_for_elitism > child_fitness_scores_for_elitism[worst_child_idx]:
                        pop[worst_child_idx] = best_overall_chromosome.copy()

            if verbose_debug_ga:
                tqdm.write(f"Gen {gen+1}/{generations} - Overall Best Fitness: {best_overall_fitness:.4f}")
            else:
                tqdm.write(f"Gen {gen+1}/{generations} - Overall Best Fitness: {best_overall_fitness:.4f}", end='\r')


    if best_overall_chromosome is None or best_overall_fitness == -float('inf'):
        if len(pop) > 0:
            final_scores = np.array(pool.map(worker_func, pop))
            if final_scores.size > 0:
                best_overall_chromosome = pop[np.argmax(final_scores)]
                best_overall_fitness = np.max(final_scores)
    
    if best_overall_chromosome is None or best_overall_fitness == -float('inf'):
        print("ERROR: GA failed to find any valid solution satisfying all constraints.")
        return None, -float('inf')
            
    print(f"\nEvolutionary search completed. Best overall combined fitness: {best_overall_fitness:.4f}")
    final_arch = decode_chromosome(best_overall_chromosome, num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga, exp_opt_values)
    
    if verbose_debug_ga:
        print("\n--- Best Architecture Details ---")
        _ = calculate_fitness(
            best_overall_chromosome, mac_budget, param_budget, constraint_type, rho0_constraint, width_mult_options, exp_opt_values,
            num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga, arch_config_params, effectiveness_fitness_weight,
            latency_budget_ms, latency_fitness_weight, verbose_debug=True
        )

    return final_arch, best_overall_fitness