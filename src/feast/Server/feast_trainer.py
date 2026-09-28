from abc import ABC, abstractmethod
import time
import numpy as np
import logging
import copy
import wandb
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from ofa.utils import flops_counter as fp
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from feast.utils.subnet_cost import subnet_macs
from feast.Client.subnet_trainer import SubnetTrainer
from feast.Server.base_server_model import wandb_run_is_active

"""
Notes:
Local bn skipped
Don't support datasets with following format: self.args.dataset.startswith("stackoverflow"):
REMOVED CERTAIN SKIPS AND IF STATEMENTS IN ADD_SUBNET FUNCTIONS FOR AGGREGATION
PROXIMAL DIST LOSS NOT IMPLEMENTED
LAYERWISE WD NOT IMPLEMENTED
"""

# Note: note filtering non-trainable parameters
def model_vector(model, req_grad=True):
    if not req_grad:
        for p in model.parameters():
            p.requires_grad = False
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)


def model_dict_to_vector(model_dict, model_copy, req_grad=True):
    model_copy.load_state_dict(model_dict)
    return model_vector(model_copy, req_grad=req_grad)


def _train_client_worker(job):
    """
    Pure per-client training worker for parallel execution.

    Creates an isolated SubnetTrainer so no mutable state is shared with other
    workers or with the main thread's self.client_trainer. The server model is
    only read (deepcopied inside SubnetTrainer.train()), never mutated here.
    NOTE: threaded execution is NOT deterministic due to interleaved RNG draws
    (np.random, torch.randperm, Mixup/CutMix). Do not use --multi_gpu for
    canonical paper runs.
    """
    trainer = SubnetTrainer(job['model'], job['device'], job['args'], teacher_model=job.get('teacher_model'))
    trainer.update_local_dataset(
        job['client_idx'],
        job['local_training_data'],
        job['local_test_data'],
        job['local_sample_number'],
    )
    trainer.set_model(
        job['model'],
        arch_bundle=job['arch_bundle'],
        client_budget=job['client_budget'],
        per_step_data=job['per_step_data'],
    )
    if job['coverage_stats'] is not None:
        trainer.set_coverage_stats(job['coverage_stats'])
    w_local = trainer.train(job['lr'], job['epochs'])
    if w_local is None:
        raise RuntimeError(f"Client {job['client_idx']} worker returned None — aborting round.")
    return w_local, trainer.get_sample_number()


class FeastTrainer(ABC):
    def __init__(
        self,
        server_model,
        dataset,
        client_trainer,
        args,
        lr_scheduler,
        wt_avg_sched_method="Uniform",
        teacher_model=None,
        start_round=0,
        best_accuracy=None,
    ):
        self.server_model = server_model
        # Dataset array format (11 elements):
        # 0: train_data_num, 1: val_data_num, 2: test_data_num
        # 3: train_data_global, 4: val_data_global, 5: test_data_global
        # 6: train_data_local_num_dict, 7: train_data_local_dict, 8: test_data_local_dict
        # 9: class_num
        # 10: bn_calibration_global (for BN reset, disjoint from val_global)
        train_data_num = dataset[0]
        val_data_num = dataset[1]
        test_data_num = dataset[2]
        train_data_global = dataset[3]
        val_data_global = dataset[4]
        test_data_global = dataset[5]
        train_data_local_num_dict = dataset[6]
        train_data_local_dict = dataset[7]
        test_data_local_dict = dataset[8]
        class_num = dataset[9]
        bn_calibration_global = dataset[10] if len(dataset) > 10 else None
        
        self.train_global = train_data_global
        self.val_global = val_data_global  # Server validation set (for periodic eval during training)
        self.test_global = test_data_global  # Official test set (for final evaluation only!)
        self.bn_calibration_global = bn_calibration_global  # NEW: Disjoint subset for BN reset
        self.train_data_num_in_total = train_data_num
        self.val_data_num_in_total = val_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.args = args
        self.start_round = start_round
        self.lr_scheduler = lr_scheduler
        self.teacher_model = teacher_model
        self.client_trainer = client_trainer
        self.sampler_args = dict()
        self.server_model.load_client_sample_counts(train_data_local_num_dict)
        self.server_model.set_top_bottom_k(self.args.top_k_maxnet, self.args.bottom_k_maxnet)
        self.sampler_args["client_per_round"] = self.args.client_num_per_round
        self.sampler_args["K"] = self.args.num_multi_archs
        self.sampler_args["diverse_subnets"] = self.args.diverse_subnets
        self.sampler_args["ps_depth_only"] = self.args.ps_depth_only

        # DeepFedNAS-specific GA search parameters, forwarded to the
        # entropy-maximizing samplers (TS_entropy_maximizer / TS_optimal_path)
        if hasattr(args, 'supernet_rho0_constraint'):
            self.sampler_args["supernet_rho0_constraint"] = self.args.supernet_rho0_constraint
        if hasattr(args, 'supernet_effectiveness_fitness_weight'):
            self.sampler_args["supernet_effectiveness_fitness_weight"] = self.args.supernet_effectiveness_fitness_weight
        if hasattr(args, 'ga_pop_size'):
            self.sampler_args["ga_pop_size"] = self.args.ga_pop_size
        if hasattr(args, 'ga_generations'):
            self.sampler_args["ga_generations"] = self.args.ga_generations
        if hasattr(args, 'ga_mutate_p'):
            self.sampler_args["ga_mutate_p"] = self.args.ga_mutate_p
        if hasattr(args, 'subnet_cache_path'):
            self.sampler_args["subnet_cache_path"] = self.args.subnet_cache_path
        # if hasattr(args, 'optimal_path_cache_path'):
        #     self.sampler_args["optimal_path_cache_path"] = self.args.optimal_path_cache_path

        self.wt_avg_sched_method = wt_avg_sched_method
        self.weighted_avg_scheduler = dict()
        self.weighted_avg_scheduler["Uniform"] = self.uniform_avg
        self.weighted_avg_scheduler[
            "maxnet_linear_all_subnet"
        ] = self.maxnet_linear_all_subnet
        self.weighted_avg_scheduler[
            "minnet_linear_all_subnet"
        ] = self.minnet_linear_all_subnet
        self.weighted_avg_scheduler[
            "maxnet_cos_all_subnet"
        ] = self.maxnet_cos_all_subnet
        self.weighted_avg_scheduler[
            "minnet_cos_all_subnet"
        ] = self.minnet_cos_all_subnet
        self.weighted_avg_scheduler[
            "maxnet_cos_all_subnet_sandwich"
        ] = self.maxnet_cos_all_subnet_sandwich
        self.weighted_avg_scheduler[
            "phased_maxnet_all_subnet"
        ] = self.phased_maxnet_all_subnet
        self.weighted_avg_scheduler[
            "multikd_phased_maxnet_all_subnet"
        ] = self.multikd_phased_maxnet_all_subnet

        # Wt Avg Scheduler Local Vars
        self.wt_avg_init = False
        self.init_maxnet_avg_wt = None
        self.mid_maxnet_avg_wt = None
        self.final_maxnet_avg_wt = None
        self.init_minnet_avg_wt = None
        self.final_minnet_avg_wt = None
        self.alpha = None
        self.weight_increment = None
        self.weight_increment1 = None
        self.weight_increment2 = None
        self.num_steps = None
        self.num_steps1 = None
        self.num_steps2 = None

        # Best Model Checkpointing
        # prev_best: global best across entire training run (never reset)
        # interval_best: best within the current 1000-round window (resets each interval)
        self.prev_best = best_accuracy if best_accuracy is not None else 0.0
        self.best_model_interval = self.args.best_model_freq
        if self.args.best_model_freq > 0:
            while self.best_model_interval <= start_round:
                self.best_model_interval += self.args.best_model_freq
        self.interval_best = 0.0

        # FedDyn alpha and server state
        self.feddyn = self.args.feddyn
        self.feddyn_alpha = args.feddyn_alpha
        self.model_param_len = 0
        if self.feddyn:
            self.model_param_len = model_vector(self.server_model.get_model_copy()).size()[
                0
            ]
        self.server_state = np.zeros(
            (self.args.client_num_in_total, self.model_param_len)
        ).astype("float32")
        self.weighted_feddyn_alpha = np.asarray(
            [
                self.train_data_local_num_dict[client_idx]
                for client_idx in range(self.args.client_num_in_total)
            ]
        )
        self.weighted_feddyn_alpha = (
            (self.weighted_feddyn_alpha / np.sum(self.weighted_feddyn_alpha))
            * self.feddyn_alpha
            * self.args.client_num_in_total
        )
        self.weighted_feddyn_alpha = (
            (self.weighted_feddyn_alpha / np.sum(self.weighted_feddyn_alpha))
            * self.feddyn_alpha
            * self.args.client_num_in_total
        )
        
        # Performance Tracking for Subnets in Cache
        self.subnet_loss_tracker = {} # Key: Subnet String/Index, Value: Last Loss

        # Sub-Supernet Communication Support
        # When use_sub_supernet is enabled, we track mappings for sparse aggregation
        self.use_sub_supernet = getattr(self.args, 'use_sub_supernet', False)
        self.sub_supernet_mappings = {}  # client_idx -> mapping dict

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        # ... (Existing implementation) ...
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.debug("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _get_model_for_client(self, client_idx, client_budget):
        """
        Get the appropriate model for a client, optionally creating a sub-supernet.

        Args:
            client_idx: Client index
            client_budget: Client's MAC budget

        Returns:
            model: Either full supernet or sub-supernet
        """
        if not self.use_sub_supernet:
            # Standard: use full supernet
            self.sub_supernet_mappings[client_idx] = 'full'
            return self.server_model.model

        # Create sub-supernet for this client's budget
        if (hasattr(self.server_model, 'subnet_cache_macs') and self.server_model.subnet_cache_macs
                and client_budget < self.server_model.subnet_cache_macs[0]):
            if hasattr(self, '_round_no_valid_subnet_count'):
                self._round_no_valid_subnet_count += 1
        sub_supernet, sub_info = self.server_model.create_sub_supernet(client_budget)

        if sub_info['is_full_supernet']:
            # No reduction possible - use full supernet
            self.sub_supernet_mappings[client_idx] = 'full'
            return self.server_model.model
        else:
            # Store mapping for sparse aggregation
            self.sub_supernet_mappings[client_idx] = sub_info['mapping']
            logging.debug(f"Client {client_idx}: Using sub-supernet "
                         f"(params: {sum(p.numel() for p in sub_supernet.parameters())/1e6:.2f}M)")
            return sub_supernet

    def _aggregate(self, w_locals, client_indexes=None):
        w_global_max_net = self.server_model.get_model_params()
        shared_param_count = dict()
        shared_param_sum = dict()
        for key, tensor in w_global_max_net.items():
            shared_param_count[key] = torch.zeros_like(tensor)
            shared_param_sum[key] = torch.zeros_like(tensor)

        # sum the local models.
        for idx, w_local in enumerate(w_locals):
            # DEBUG: Check w_local for NaNs
            if w_local is None:
                logging.error("[DEBUG] Encountered None w_local in aggregation!")
                continue

            has_nan = False
            for k, v in w_local.state_dict().items():
                if torch.isnan(v).any():
                    has_nan = True
                    logging.error(f"[DEBUG] NaN found in w_local params for key {k}")
                    break
            if has_nan:
                logging.error("[DEBUG] Skipping aggregation for corrupted w_local")
                continue

            # Check if this client used sub-supernet
            client_idx = client_indexes[idx] if client_indexes is not None else None
            mapping = self.sub_supernet_mappings.get(client_idx) if client_idx is not None else None

            if mapping is not None and mapping != 'full':
                # Sub-supernet: use sparse aggregation
                self.server_model.add_sub_supernet(
                    shared_param_sum, shared_param_count,
                    w_local.state_dict(), mapping,
                    weight=getattr(w_local, 'avg_weight', 1.0)
                )
            else:
                # Full supernet: use standard aggregation
                self.server_model.add_subnet(shared_param_sum, shared_param_count, w_local)

        # DEBUG: Check shared accumulators for NaNs
        for k, v in shared_param_sum.items():
            if torch.isnan(v).any():
                logging.error(f"[DEBUG] NaN found in shared_param_sum for key {k} BEFORE averaging")
        

        # [Aggregate Logic]
        # shared_weights = zero tensors
        # update shared_weights from participating subnets by summing
        # w_supernet_{t+1} = shared_weights * 1/max(number_of_overlaps, 1) + (shared_weights is zero) w_supernet_{t}
        for key in w_global_max_net:
            w_global_max_net[key] = (
                shared_param_sum[key]
                * (
                    1.0
                    / (shared_param_count[key] + (shared_param_count[key] == 0).int())
                )
                + (shared_param_count[key] == 0) * w_global_max_net[key]
            )
        return w_global_max_net

    def train(self):
        # Log Client Budgets to WandB
        # Client Budgets are saved to client_budgets.csv in generic_server_model.py
        pass

        if self.args.ckpt_subnets is None:
            self.args.ckpt_subnets = []
            for subnet_id in self.args.diverse_subnets:
                self.args.ckpt_subnets.append(self.args.diverse_subnets[subnet_id])
        for round_idx in range(self.start_round, self.args.comm_round):
            if self.args.best_model_freq > 0 and round_idx >= self.best_model_interval:
                self.best_model_interval += self.args.best_model_freq
                self.interval_best = 0.0
            self.train_one_round(round_idx)
        if self.args.dry_run:
            return
        if self.args.wandb_watch:
            self.server_model.wandb_pass()
        if self.args.model_checkpoint_freq > 0:
            self.server_model.save(
                "finished_checkpoint_data.pt",
                current_round=self.args.comm_round,
                best_accuracy=self.prev_best
            )

    def train_one_round(self, round_num):
        round_start_time = time.time()
        self.server_model.model.train()
        client_indexes = self._client_sampling(
            round_num, self.args.client_num_in_total, self.args.client_num_per_round
        )

        logging.info(f"=== Round {round_num}/{self.args.comm_round-1} | Clients: {client_indexes.tolist()} ===")

        self.server_model.set_cli_indices(client_indexes)
        self.server_model.update_sample()
        avg_weights = self.weighted_avg_scheduler[self.wt_avg_sched_method](round_num)

        w_locals = []
        training_samples = []
        self._round_no_valid_subnet_count = 0  # Track repeated warnings per round

        client_devices = getattr(self.args, 'client_devices', [self.client_trainer.device])
        client_jobs = []

        for idx, client_idx in enumerate(client_indexes):
            # self.client_trainer.client_idx = client_idx # Replaced by update_local_dataset
            self.client_trainer.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            
            # 1. Fetch Client Budget
            client_budget = self.server_model.client_mac_budgets.get(client_idx, float('inf'))

            # 2. Determine Architectures for "Inverse KD Sandwich"
            if self.args.training_strategy == 'inverse_kd_sandwich':
                # A. Min Arch: Smallest in Cache (User Request)
                min_arch_config, min_macs = self.server_model.get_min_cached_subnet_info()
                min_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(min_arch_config)
                
                # B. Max Arch: Local Max (largest feasible for this client)
                max_arch_config, max_macs = self.server_model.get_client_local_max_info(client_idx)
                
                # Check for Degeneracy: Min ~ Max
                # Use a small epsilon for float comparison safety
                if abs(min_macs - max_macs) < 1e-4:
                     # Case 1: Budget is so low that Local Max is essentially the Min.
                     # Action: Train ONLY Min. Skip Max and Random.
                     arch_bundle = {
                        'min': {**min_arch_config, **min_subnet_kwargs}
                     }
                else:
                    max_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(max_arch_config)
                    
                    
                    # C. Random Arch: Strictly Constrained (Min < Random < Max)
                    random_arch_config, random_idx = self.server_model.random_strictly_constrained_cached_subnet_arch(min_macs, max_macs)
                    
                    if random_arch_config:
                        # Case 2: Standard Sandwich (Min, Max, Random)
                        random_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(random_arch_config)
                        arch_bundle = {
                            'min': {**min_arch_config, **min_subnet_kwargs},
                            'max': {**max_arch_config, **max_subnet_kwargs},
                            'random': {**random_arch_config, **random_subnet_kwargs}
                        }
                    else:
                        # Case 3: No subnet exists strictly between Min and Max.
                        # Action: Train Min and Max. Skip Random.
                        arch_bundle = {
                            'min': {**min_arch_config, **min_subnet_kwargs},
                            'max': {**max_arch_config, **max_subnet_kwargs}
                        }

                # Prepare per_step_data if per_step_random is enabled
                per_step_data = None
                if getattr(self.args, 'per_step_random', False) and 'random' in arch_bundle:
                    # Filter subnet cache to subnets strictly between min and max
                    filtered_cache = []
                    filtered_macs = []
                    filtered_coverage = {}
                    for idx, (subnet, macs) in enumerate(zip(self.server_model.subnet_cache, self.server_model.subnet_cache_macs)):
                        if min_macs < macs < max_macs:
                            filtered_cache.append(subnet)
                            filtered_macs.append(macs)
                            if hasattr(self.server_model, 'subnet_coverage_map'):
                                filtered_coverage[len(filtered_cache) - 1] = self.server_model.subnet_coverage_map.get(idx, 1.0)
                    
                    per_step_data = {
                        'filtered_cache': filtered_cache,
                        'filtered_macs': filtered_macs,
                        'coverage_map': filtered_coverage,
                        'min_macs': min_macs,
                        'max_macs': max_macs,
                        'arch_to_subnet_kwargs': self.server_model.arch_to_subnet_kwargs,  # Pass method reference
                    }

                # Get model (full or sub-supernet based on --use_sub_supernet flag)
                model_for_client = self._get_model_for_client(client_idx, client_budget)
                self.client_trainer.set_model(
                    model_for_client,
                    client_budget=client_budget,
                    arch_bundle=arch_bundle,
                    per_step_data=per_step_data
                )

            # --- E0: Min Only (All clients train min subnet) ---
            elif self.args.training_strategy == 'min_only':
                min_arch_config, min_macs = self.server_model.get_min_cached_subnet_info()
                min_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(min_arch_config)

                arch_bundle = {
                    'min': {**min_arch_config, **min_subnet_kwargs}
                }
                # Get model (full or sub-supernet based on --use_sub_supernet flag)
                model_for_client = self._get_model_for_client(client_idx, client_budget)
                # No sandwich mode - single subnet training
                self.client_trainer.set_model(
                    model_for_client,
                    client_budget=client_budget,
                    arch_bundle=arch_bundle
                )
            
            # --- E1: Local Max Only (Each client trains only its local max) ---
            elif self.args.training_strategy == 'local_max_only':
                max_arch_config, max_macs = self.server_model.get_client_local_max_info(client_idx)
                max_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(max_arch_config)
                
                arch_bundle = {
                    'max': {**max_arch_config, **max_subnet_kwargs}
                }
                # Get model (full or sub-supernet based on --use_sub_supernet flag)
                model_for_client = self._get_model_for_client(client_idx, client_budget)
                # No sandwich mode - single subnet training
                self.client_trainer.set_model(
                    model_for_client,
                    client_budget=client_budget,
                    arch_bundle=arch_bundle
                )

            # --- E2: FEAST Per-Step Min/Random/Max (Max→Min, Max→Rand KD) ---
            elif self.args.training_strategy == 'feast':
                # A. Min Arch
                min_arch_config, min_macs = self.server_model.get_min_cached_subnet_info()
                min_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(min_arch_config)
                
                # B. Max Arch (Local)
                max_arch_config, max_macs = self.server_model.get_client_local_max_info(client_idx)
                
                if abs(min_macs - max_macs) < 1e-4:
                    # Degenerate case: only train min
                    arch_bundle = {
                        'min': {**min_arch_config, **min_subnet_kwargs}
                    }
                else:
                    max_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(max_arch_config)

                    # C. Random Arch (ablation: --disable_random_step drops this entirely)
                    random_arch_config = None
                    if not getattr(self.args, 'disable_random_step', False):
                        random_arch_config, random_idx = self.server_model.random_strictly_constrained_cached_subnet_arch(min_macs, max_macs)

                    if random_arch_config:
                        random_subnet_kwargs = self.server_model.arch_to_subnet_kwargs(random_arch_config)
                        arch_bundle = {
                            'min': {**min_arch_config, **min_subnet_kwargs},
                            'max': {**max_arch_config, **max_subnet_kwargs},
                            'random': {**random_arch_config, **random_subnet_kwargs}
                        }
                    else:
                        arch_bundle = {
                            'min': {**min_arch_config, **min_subnet_kwargs},
                            'max': {**max_arch_config, **max_subnet_kwargs}
                        }
                
                # Prepare per_step_data if per_step_random is enabled
                per_step_data = None
                if getattr(self.args, 'per_step_random', False) and 'random' in arch_bundle:
                    # Filter subnet cache to subnets strictly between min and max
                    filtered_cache = []
                    filtered_macs = []
                    filtered_coverage = {}
                    for idx, (subnet, macs) in enumerate(zip(self.server_model.subnet_cache, self.server_model.subnet_cache_macs)):
                        if min_macs < macs < max_macs:
                            filtered_cache.append(subnet)
                            filtered_macs.append(macs)
                            if hasattr(self.server_model, 'subnet_coverage_map'):
                                filtered_coverage[len(filtered_cache) - 1] = self.server_model.subnet_coverage_map.get(idx, 1.0)
                    
                    per_step_data = {
                        'filtered_cache': filtered_cache,
                        'filtered_macs': filtered_macs,
                        'coverage_map': filtered_coverage,
                        'min_macs': min_macs,
                        'max_macs': max_macs,
                        'arch_to_subnet_kwargs': self.server_model.arch_to_subnet_kwargs,
                    }

                # Get model (full or sub-supernet based on --use_sub_supernet flag)
                model_for_client = self._get_model_for_client(client_idx, client_budget)
                self.client_trainer.set_model(
                    model_for_client,
                    client_budget=client_budget,
                    arch_bundle=arch_bundle,
                    per_step_data=per_step_data
                )

            else:
                 # Standard: one subnet per client per round, chosen by --subnet_dist_type
                 # (TS_all_random / TS_optimal_path / TS_all_random_constrained / ...).
                 # sample_subnet() also updates cli_subnet_track, which update_sample()
                 # above reads to pick this round's largest_subnet_min_idx / smallest_subnet_min_idx
                 # for the MaxNet weighting schedule.
                 model_for_client = self.server_model.sample_subnet(
                     round_num, client_idx, idx, self.sampler_args
                 )
                 model_for_client.set_avg_wt(avg_weights[idx])
                 self.client_trainer.set_model(model_for_client, client_budget=client_budget)

            # 3. Model Updates & Training
            cur_lr = self.args.lr
            if self.lr_scheduler is not None:
                cur_lr = self.lr_scheduler.get_lr(round_num)
            
            # Pass coverage stats for adaptive KD ratio computation (per-client)
            if hasattr(self.client_trainer, 'set_coverage_stats'):
                # Determines local_max_mac for this client
                # Priority 1: Use max_macs found during bundle creation (most accurate)
                # Priority 2: Extract from 'max' in arch_bundle (re-calc)
                # Priority 3: Global Max (Fallback)
                
                local_max_mac_for_cov = None
                
                if 'max_macs' in locals() and max_macs is not None:
                     local_max_mac_for_cov = max_macs
                elif hasattr(self.client_trainer, 'arch_bundle') and self.client_trainer.arch_bundle is not None and 'max' in self.client_trainer.arch_bundle:
                    max_config = self.client_trainer.arch_bundle['max']
                    local_max_mac_for_cov, _ = subnet_macs(
                        depth_vec=max_config['d'],
                        exp_vec=max_config['e'],
                        w_indices=max_config['w_indices'],
                        width_mult_options=self.server_model.arch_params['width_multiplier_choices'],
                        arch_config_params=self.server_model.arch_params,
                    )
                
                coverage_stats = self.server_model.compute_subnet_coverage(local_max_mac=local_max_mac_for_cov)
                
                # Optionally override with the sampled random subnet's specific coverage value
                if self.args.adaptive_kd_specific_rand and 'random_idx' in locals() and random_idx is not None:
                     if hasattr(self.server_model, 'subnet_coverage_map'):
                         specific_cov = self.server_model.subnet_coverage_map.get(random_idx)
                         if specific_cov is not None:
                             coverage_stats['coverage_rand'] = specific_cov
                             # logging.debug(f"Using specific coverage for random subnet {random_idx}: {specific_cov:.4f}")

                self.client_trainer.set_coverage_stats(coverage_stats)

            if getattr(self.args, 'multi_gpu', False):
                # --- PARALLEL PATH: queue immutable job package ---
                # Sequential setup above is complete; capture state from shared trainer.
                # Worker deepcopies job['model'] inside train(). For full-supernet mode
                # job['model'] is self.server_model.model — read-only from the worker since
                # no server-side mutation occurs until all workers finish.
                client_jobs.append({
                    'job_idx': len(client_jobs),
                    'client_idx': client_idx,
                    'model': self.client_trainer.client_model,
                    'arch_bundle': self.client_trainer.arch_bundle,
                    'per_step_data': self.client_trainer.per_step_data,
                    'coverage_stats': getattr(self.client_trainer, 'coverage_stats', None),
                    'local_training_data': self.client_trainer.local_training_data,
                    'local_test_data': self.client_trainer.local_test_data,
                    'local_sample_number': self.client_trainer.local_sample_number,
                    'client_budget': client_budget,
                    'lr': cur_lr,
                    'epochs': self.args.epochs,
                    # len(client_jobs) before append gives job indices 0, 1, 2, ...
                    'device': client_devices[len(client_jobs) % len(client_devices)],
                    'args': self.args,
                    'teacher_model': self.teacher_model,
                })
            else:
                # --- SEQUENTIAL PATH: original behaviour, unchanged ---
                w_local = self.client_trainer.train(
                    cur_lr,
                    self.args.epochs
                )

                # Ensure avg_weight is set
                if not hasattr(w_local, 'avg_weight') or w_local.avg_weight is None:
                    w_local.avg_weight = 1.0

                # Recover metrics for WandB logging
                if hasattr(w_local, 'training_metrics') and w_local.training_metrics:
                    for role, data in w_local.training_metrics.items():
                        arch = data.get('arch')
                        loss = data.get('loss')
                        if arch and loss is not None:
                            if self.server_model.subnet_cache:
                                for i, cached in enumerate(self.server_model.subnet_cache):
                                    import numpy as np
                                    d_match = (cached.get('d') == arch.get('d'))
                                    w_match = (cached.get('w_indices') == arch.get('w_indices'))
                                    e_match = False
                                    if d_match and w_match:
                                        c_e = cached.get('e', [])
                                        a_e = arch.get('e', [])
                                        if len(c_e) == len(a_e):
                                            e_match = np.allclose(c_e, a_e, atol=1e-5)
                                    if d_match and w_match and e_match:
                                        self.subnet_loss_tracker[i] = loss
                                        break

                training_samples.append(self.client_trainer.get_sample_number())

                if w_local is None:
                    logging.error(f"[DEBUG] Client {client_idx} returned None model!")
                else:
                    for k, v in w_local.state_dict().items():
                        if torch.isnan(v).any():
                            logging.error(f"[DEBUG] Client {client_idx} returned NaN weights in {k}")

                w_locals.append(w_local)

        if getattr(self.args, 'multi_gpu', False) and client_jobs:
            # --- Parallel execution ---
            n_workers = len(client_devices)
            logging.info(
                f"[multi_gpu] Round {round_num}: submitting {len(client_jobs)} client jobs "
                f"to {n_workers} worker(s) on devices {[str(d) for d in client_devices]}"
            )
            results_map = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(_train_client_worker, job): job['job_idx']
                    for job in client_jobs
                }
                for future in as_completed(future_to_idx):
                    job_idx = future_to_idx[future]
                    exc = future.exception()
                    if exc is not None:
                        raise RuntimeError(
                            f"Round {round_num}: client worker job_idx={job_idx} raised: {exc}"
                        ) from exc
                    results_map[job_idx] = future.result()

            # Post-processing in original client order
            for job in client_jobs:
                job_idx = job['job_idx']
                client_idx = job['client_idx']
                w_local, sample_count = results_map[job_idx]

                if not hasattr(w_local, 'avg_weight') or w_local.avg_weight is None:
                    w_local.avg_weight = 1.0

                if hasattr(w_local, 'training_metrics') and w_local.training_metrics:
                    for role, data in w_local.training_metrics.items():
                        arch = data.get('arch')
                        loss = data.get('loss')
                        if arch and loss is not None:
                            if self.server_model.subnet_cache:
                                for i, cached in enumerate(self.server_model.subnet_cache):
                                    import numpy as np
                                    d_match = (cached.get('d') == arch.get('d'))
                                    w_match = (cached.get('w_indices') == arch.get('w_indices'))
                                    e_match = False
                                    if d_match and w_match:
                                        c_e = cached.get('e', [])
                                        a_e = arch.get('e', [])
                                        if len(c_e) == len(a_e):
                                            e_match = np.allclose(c_e, a_e, atol=1e-5)
                                    if d_match and w_match and e_match:
                                        self.subnet_loss_tracker[i] = loss
                                        break

                training_samples.append(sample_count)

                nan_keys = [k for k, v in w_local.state_dict().items() if torch.isnan(v).any()]
                if nan_keys:
                    raise RuntimeError(
                        f"Round {round_num}: client {client_idx} returned NaN weights in: {nan_keys}"
                    )

                w_locals.append(w_local)

        # Aggregate
        if self.args.weight_dataset:
            total_training_samples = sum(training_samples)
            for i, w_loc in enumerate(w_locals):
                w_loc.avg_weight = w_loc.avg_weight * (float(training_samples[i]) / total_training_samples)
                logging.debug(f"[Agg] Client {i} Weight: {w_loc.avg_weight}")
            logging.debug(f"[Agg] Total Samples: {total_training_samples}, Individual: {training_samples}")

        supernet_aggregate = self._aggregate(w_locals, client_indexes)
        self.server_model.set_model_params(supernet_aggregate)

        # Clear sub-supernet mappings for next round
        self.sub_supernet_mappings.clear()
        
        # Append this round's per-subnet training loss to a CSV (one column per
        # round, one row per cached subnet), tracked in wandb.run.dir instead of
        # as wandb plots.
        if wandb_run_is_active() and self.server_model.subnet_cache:
             import csv
             import os
             
             csv_path = os.path.join(wandb.run.dir, "subnet_losses.csv")
             
             # Prepare current round data
             current_round_losses = []
             # Headers for standard columns
             subnet_indices = []
             subnet_macs_list = []
             
             for idx, cached_arch in enumerate(self.server_model.subnet_cache):
                 subnet_indices.append(idx)
                 macs = 0
                 if hasattr(self.server_model, 'subnet_cache_macs'):
                     macs = self.server_model.subnet_cache_macs[idx]
                 subnet_macs_list.append(macs)
                 
                 # Get last known loss (tracker persists values)
                 # Use "N/A" if never seen
                 val = self.subnet_loss_tracker.get(idx, "N/A")
                 current_round_losses.append(val)
             
             # Read/Write Logic for CSV (Column Appending)
             rows = []
             headers = ["Subnet Index", "MACs"]
             
             if os.path.exists(csv_path):
                 with open(csv_path, 'r') as f:
                     reader = csv.reader(f)
                     existing_rows = list(reader)
                     if existing_rows:
                         headers = existing_rows[0]
                         rows = existing_rows[1:]
             else:
                 # Initialize rows if file doesn't exist
                 for i in range(len(subnet_indices)):
                     rows.append([str(subnet_indices[i]), f"{subnet_macs_list[i]}"])

             # Append new column header
             headers.append(f"Loss_R{round_num}")
             
             # Append new data to each row
             # Ensure rows align with cache index (rows[i] corresponds to index i)
             # Note: This assumes cache size/order doesn't change, which is true for fixed cache.
             if len(rows) == len(current_round_losses):
                 for i in range(len(rows)):
                     val = current_round_losses[i]
                     # Format if float
                     if isinstance(val, (float, int)):
                         val_str = f"{val:.2f}"
                     else:
                         val_str = str(val)
                     rows[i].append(val_str)
             else:
                 # Should not happen if cache is fixed
                 print(f"[Warning] Subnet cache size mismatch! CSV Rows: {len(rows)}, Cache: {len(current_round_losses)}")
                 
             # Write back
             with open(csv_path, 'w', newline='') as f:
                 writer = csv.writer(f)
                 writer.writerow(headers)
                 writer.writerows(rows)
                 
             # Force upload to WandB for immediate visibility
             wandb.save(csv_path, base_path=wandb.run.dir)
             
        if self.feddyn: # Safely check feddyn
            pass 

        if self.args.wandb_watch and round_idx % self.args.wandb_watch_freq == 0:
            self.server_model.wandb_pass()

        # test results
        # at last round
        if round_num == self.args.comm_round - 1:
            if self.args.efficient_test:
                self._efficient_local_test_on_all_clients(round_num)
            else:
                self._local_test_on_all_clients(round_num)
            # Evaluate global max on supporting clients if enabled
            self._eval_global_max_on_supporters(round_num)
        # per {frequency_of_the_test} round
        elif round_num % self.args.frequency_of_the_test == 0:
            if self.args.efficient_test:
                (_, subnet_test_acc_map,) = self._efficient_local_test_on_all_clients(
                    round_num
                )
            else:
                _, subnet_test_acc_map = self._local_test_on_all_clients(round_num)
            # Evaluate global max on supporting clients if enabled
            self._eval_global_max_on_supporters(round_num)
            # Reuse cached metrics from _efficient_local_test_on_all_clients instead of re-evaluating
            # This avoids redundant BN resets for the same subnets
            if subnet_test_acc_map:
                mean_acc = sum(subnet_test_acc_map.values()) / len(subnet_test_acc_map)
            else:
                mean_acc = 0.0
            if self.args.dataset == 'ptb':
                wandb.log(
                    {f"Test/Mean/PPL": mean_acc, "round": round_num,}
                )
                if mean_acc < self.prev_best:
                    self.prev_best = mean_acc
                    self.server_model.save(
                        "best_checkpoint_supernet.pt",
                        current_round=round_num,
                        best_accuracy=self.prev_best
                    )
                if self.args.best_model_freq > 0 and (mean_acc < self.interval_best or self.interval_best == 0.0):
                    self.interval_best = mean_acc
                    self.server_model.save(
                        f"best_checkpoint_supernet_{self.best_model_interval}.pt",
                        current_round=round_num,
                        best_accuracy=self.interval_best
                    )
            else:
                # Label correctly based on which dataset was used for evaluation
                eval_metric_prefix = "Val" if (hasattr(self, 'val_global') and self.val_global is not None) else "Test"
                wandb.log(
                    {f"{eval_metric_prefix}/Mean/Acc": mean_acc, "round": round_num,}
                )
                if mean_acc > self.prev_best:
                    self.prev_best = mean_acc
                    self.server_model.save(
                        "best_checkpoint_supernet.pt",
                        current_round=round_num,
                        best_accuracy=self.prev_best
                    )
                if self.args.best_model_freq > 0 and mean_acc > self.interval_best:
                    self.interval_best = mean_acc
                    self.server_model.save(
                        f"best_checkpoint_supernet_{self.best_model_interval}.pt",
                        current_round=round_num,
                        best_accuracy=self.interval_best
                    )
        # Checkpointing
        if (self.args.model_checkpoint_freq > 0
                and round_num > 0
                and round_num % self.args.model_checkpoint_freq == 0):
            self.server_model.save(
                f"finished_checkpoint_data_{round_num}.pt",
                current_round=round_num,
                best_accuracy=self.prev_best
            )
        # Saving after each 'frequency_of_the_test' rounds
        latest_model_name = f"latest_round_model.pt"
        self.server_model.save(
             latest_model_name,
             current_round=round_num,
             best_accuracy=self.prev_best
        )
        round_elapsed = time.time() - round_start_time
        # Log warning count for suppressed sub-supernet warnings
        warn_suffix = ""
        if hasattr(self, '_round_no_valid_subnet_count') and self._round_no_valid_subnet_count > 0:
            warn_suffix = f" | {self._round_no_valid_subnet_count} clients below min cache"
        logging.info(f"--- Round {round_num} done | {round_elapsed:.1f}s | best={self.prev_best*100:.2f}%{warn_suffix} ---")

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################ Evaluating Diverse Subnets on Clients (Round {}) ################".format(round_idx))

        avg_subnet_train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
        }

        avg_subnet_test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
        }
        subnet_test_acc_map = dict()
        subnet_train_acc_map = dict()
        for subnet_id in self.args.diverse_subnets:
            subnet_info = self.args.diverse_subnets[subnet_id]
            train_metrics = {
                "num_samples": [],
                "num_correct": [],
                "losses": [],
            }

            test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            for client_idx in range(self.args.client_num_in_total):
                """
                Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
                the training client number is larger than the testing client number
                """
                if self.test_data_local_dict[client_idx] is None:
                    continue

                self.client_trainer.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # set model first
                self.client_trainer.set_test_model(
                    self.server_model.get_subnet(
                        **self.args.diverse_subnets[subnet_id]
                    ),
                )

                # train data
                train_local_metrics = self.client_trainer.local_test(False)
                train_metrics["num_samples"].append(
                    copy.deepcopy(train_local_metrics["test_total"])
                )
                train_metrics["num_correct"].append(
                    copy.deepcopy(train_local_metrics["test_correct"])
                )
                train_metrics["losses"].append(
                    copy.deepcopy(train_local_metrics["test_loss"])
                )
                if self.args.verbose_test:
                    print("train stats", client_idx, train_local_metrics)

                # test data
                test_local_metrics = self.client_trainer.local_test(True)
                test_metrics["num_samples"].append(
                    copy.deepcopy(test_local_metrics["test_total"])
                )
                test_metrics["num_correct"].append(
                    copy.deepcopy(test_local_metrics["test_correct"])
                )
                test_metrics["losses"].append(
                    copy.deepcopy(test_local_metrics["test_loss"])
                )
                if self.args.verbose_test:
                    print("test stats", client_idx, test_local_metrics)

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_metrics["num_correct"]) / sum(
                train_metrics["num_samples"]
            )
            train_loss = sum(train_metrics["losses"]) / sum(
                train_metrics["num_samples"]
            )
            avg_subnet_train_metrics["num_correct"].append(train_acc)
            avg_subnet_train_metrics["losses"].append(train_loss)
            avg_subnet_train_metrics["num_samples"].append(1)

            # test on test dataset
            test_acc = sum(test_metrics["num_correct"]) / sum(
                test_metrics["num_samples"]
            )
            test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
            avg_subnet_test_metrics["num_correct"].append(test_acc)
            avg_subnet_test_metrics["losses"].append(test_loss)
            avg_subnet_test_metrics["num_samples"].append(1)

            stats = {
                "training_acc": train_acc,
                "training_loss": train_loss,
                "subnet": subnet_info,
            }

            wandb_subnet_log = dict()
            if "d" in subnet_info:
                wandb_subnet_log["d"] = subnet_info["d"]
            if "e" in subnet_info:
                e_list = subnet_info["e"]
                # Check if the list is not empty and all elements are equal to the first one
                if e_list and all(x == e_list[0] for x in e_list):
                    # If so, log only the single value
                    wandb_subnet_log["e"] = e_list[0]
                else:
                    # Otherwise, log the full list as before
                    wandb_subnet_log["e"] = e_list
            if "w_indices" in subnet_info:
                wandb_subnet_log["w"] = subnet_info["w_indices"]

            wandb.log(
                {f"Train/{wandb_subnet_log}/Acc": train_acc, "round": round_idx,}
            )
            wandb.log(
                {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,}
            )

            if self.args.verbose:
                logging.info(stats)

            subnet_train_acc_map[subnet_id] = stats["training_acc"]
            stats = {
                "test_acc": test_acc,
                "test_loss": test_loss,
                "subnet": subnet_info,
            }
            wandb.log(
                {f"Test/{wandb_subnet_log}/Acc": test_acc, "round": round_idx}
            )
            wandb.log(
                {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,}
            )
            logging.info(stats)
            subnet_test_acc_map[subnet_id] = stats["test_acc"]

        final_train_acc = sum(avg_subnet_train_metrics["num_correct"]) / sum(
            avg_subnet_train_metrics["num_samples"]
        )
        final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
            avg_subnet_train_metrics["num_samples"]
        )
        # test on test dataset
        final_test_acc = sum(avg_subnet_test_metrics["num_correct"]) / sum(
            avg_subnet_test_metrics["num_samples"]
        )
        final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
            avg_subnet_test_metrics["num_samples"]
        )
        final_stats = {
            "final_training_acc": train_acc,
            "final_training_loss": train_loss,
        }
        wandb.log({"Train/Acc": final_train_acc, "round": round_idx})
        wandb.log(
            {"Train/Loss": final_train_loss, "round": round_idx},
        )
        if self.args.verbose:
            logging.info(final_stats)

        final_stats = {
            "final_test_acc": test_acc,
            "final_test_loss": test_loss,
        }
        wandb.log({f"Test/Acc": final_test_acc, "round": round_idx})
        wandb.log({f"Test/Loss": final_test_loss, "round": round_idx})
        logging.info(final_stats)
        return subnet_train_acc_map, subnet_test_acc_map

    def _get_formatted_subnet_str(self, subnet_info):
        """Formats the subnet dictionary for cleaner logging."""
        import copy
        # Make a copy to avoid modifying the original args
        formatted_info = copy.deepcopy(subnet_info)
        
        # Check if 'e' is a list with all identical elements
        if 'e' in formatted_info and isinstance(formatted_info['e'], list):
            e_list = formatted_info['e']
            if len(e_list) > 1 and len(set(e_list)) == 1:
                formatted_info['e'] = e_list[0] # Replace list with single value
        
        # You could add similar logic for 'd' or 'w_indices' if needed
        # if 'd' in formatted_info and isinstance(formatted_info['d'], list):
        # ...
        
        return str(formatted_info)

    def _eval_global_max_on_supporters(self, round_idx):
        """
        Evaluate global max subnet only on supporting clients' training data.
        This shows how well the global max performs on data from clients that actually train it.
        """
        if not getattr(self.args, 'eval_supporters_only', False):
            return
        
        # Get global max subnet config from cache (last in sorted cache = largest)
        if not hasattr(self.server_model, 'subnet_cache') or not self.server_model.subnet_cache:
            logging.warning("[GlobalMax Supporters] No subnet cache available, skipping.")
            return
        
        global_max_config = self.server_model.subnet_cache[-1]  # Last = largest (cache is sorted by MACs)
        
        if global_max_config is None:
            return
        
        # Get global max MAC
        global_max_mac, _ = subnet_macs(
            depth_vec=global_max_config['d'],
            exp_vec=global_max_config['e'],
            w_indices=global_max_config['w_indices'],
            width_mult_options=self.server_model.arch_params['width_multiplier_choices'],
            arch_config_params=self.server_model.arch_params,
        )
        
        # Filter clients that can support this subnet
        supporter_clients = [
            cid for cid, budget in self.server_model.client_mac_budgets.items()
            if budget >= global_max_mac
        ]
        
        if not supporter_clients:
            logging.info(f"[GlobalMax Supporters] No clients can support global max (MAC={global_max_mac/1e6:.1f}M)")
            return
        
        logging.info(f"[GlobalMax Supporters] Evaluating on {len(supporter_clients)} supporting clients (MAC={global_max_mac/1e6:.1f}M)")
        
        # Set global max as test model
        self.client_trainer.set_test_model(
            self.server_model.get_subnet(**global_max_config),
        )
        
        # Evaluate on each supporter's training data
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        for client_idx in supporter_clients:
            self.client_trainer.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx] if client_idx in self.test_data_local_dict else None,
                self.train_data_local_num_dict[client_idx],
            )
            metrics = self.client_trainer.local_test(use_test_set=False)  # Use training data
            total_correct += metrics.get("test_correct", 0)
            total_samples += metrics.get("test_total", 0)
            # Note: test_loss is already cumulative (sum of loss * batch_size for each batch)
            total_loss += metrics.get("test_loss", 0)
        
        if total_samples > 0:
            supporters_acc = total_correct / total_samples
            supporters_loss = total_loss / total_samples
            
            logging.info(f"[GlobalMax Supporters] Train Acc: {supporters_acc:.4f}, Train Loss: {supporters_loss:.4f}")
            wandb.log({
                "Train/GlobalMax/SupportersOnly/Acc": supporters_acc,
                "Train/GlobalMax/SupportersOnly/Loss": supporters_loss,
                "round": round_idx,
            })

    def _efficient_local_test_on_all_clients(self, round_idx):

        logging.info(f"--- Eval Round {round_idx} ---")
        eval_start_time = time.time()

        import torch

        # Determine which evaluation set to use: validation (held-out) or test (if no validation)
        # During training, we should use val_global to avoid contaminating the test set
        if hasattr(self, 'val_global') and self.val_global is not None:
            eval_data = self.val_global
            eval_label = "val"  # Use val_acc, val_loss labels
            full_val_size = self.val_data_num_in_total
        else:
            eval_data = self.test_global
            eval_label = "test"  # Fallback: use test_acc, test_loss labels
            full_val_size = len(eval_data.dataset) if hasattr(eval_data, 'dataset') else 0
        
        # --- Sample validation set if eval_sample_size < 1.0 for faster periodic evaluation ---
        eval_sample_size = getattr(self.args, 'eval_sample_size', 1.0)
        if eval_sample_size < 1.0 and eval_sample_size > 0:
            n_samples = len(eval_data.dataset)
            subset_size = max(int(n_samples * eval_sample_size), self.args.batch_size)
            
            import torch
            g = torch.Generator()
            g.manual_seed(42 + round_idx)  # Reproducible per round
            rand_indices = torch.randperm(n_samples, generator=g).tolist()[:subset_size]
            sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_indices)
            eval_data = torch.utils.data.DataLoader(
                eval_data.dataset,
                batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 64,
                sampler=sub_sampler,
                num_workers=4,
            )
            logging.debug(f"Using {subset_size}/{full_val_size} samples ({eval_sample_size*100:.0f}%) from {eval_label} set")
        else:
            logging.debug(f"Using full {eval_label} set ({full_val_size} samples)")

        avg_subnet_train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "ppl": [],
            "losses": [],
        }

        avg_subnet_test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "ppl": [],
            "losses": [],
        }

        subnet_train_acc_map = dict()
        subnet_test_acc_map = dict()

        for subnet_id in self.args.diverse_subnets:
            subnet_info = self.args.diverse_subnets[subnet_id]
            # set model first
            self.client_trainer.set_test_model(
                self.server_model.get_subnet(**subnet_info),
            )
            
            # --- Global BN Reset (Efficient): Once per subnet using global training data ---
            if self.args.reset_bn_stats or self.args.reset_bn_stats_test:
                bn_start_time = time.time()
                
                # Get the test model and unwrap if needed
                target_model = self.client_trainer.test_model
                model_to_reset = target_model.get_model() if hasattr(target_model, 'get_model') else target_model
                
                # --- FL-Compliant BN Calibration ---
                # Use dedicated calibration set (disjoint from validation) if available
                # Otherwise fall back to train_global for backwards compatibility
                import torch
                
                if self.bn_calibration_global is not None:
                    bn_loader = self.bn_calibration_global
                    subset_size = len(self.bn_calibration_global.dataset)
                else:
                    n_global_samples = len(self.train_global.dataset)
                    subset_size = int(self.args.reset_bn_sample_size * n_global_samples)
                    subset_size = max(subset_size, self.args.batch_size)

                    g = torch.Generator()
                    g.manual_seed(42 + round_idx)
                    rand_indices = torch.randperm(n_global_samples, generator=g).tolist()[:subset_size]

                    bn_subset = torch.utils.data.Subset(self.train_global.dataset, rand_indices)
                    bn_loader = torch.utils.data.DataLoader(
                        bn_subset,
                        batch_size=self.args.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                    )
                    logging.warning(f"[BN Reset] Fallback: Using train_global subset ({subset_size} samples)")

                # Ensure model is on GPU
                try:
                    p_device = next(model_to_reset.parameters()).device
                    if p_device.type == 'cpu' and torch.cuda.is_available():
                        model_to_reset = model_to_reset.to(torch.device("cuda"))
                except Exception:
                    pass

                set_running_statistics(model_to_reset, bn_loader)

                bn_elapsed = time.time() - bn_start_time
                logging.debug(f"[BN Reset] Subnet {subnet_id}: {subset_size} samples, {bn_elapsed:.2f}s")
            
            if not self.args.skip_train_test:
                # --- OPTIMIZED: Single evaluation on global training set ---
                # Instead of looping through 100 clients, evaluate once on train_global
                train_metrics = {"num_samples": 0, "num_correct": 0, "ppl": 0, "losses": 0}
                
                # Get the test model for evaluation
                target_model = self.client_trainer.test_model
                model_to_eval = target_model.get_model() if hasattr(target_model, 'get_model') else target_model
                model_to_eval.eval()
                
                device = next(model_to_eval.parameters()).device
                # Ensure model is on GPU if available (fixes CPU evaluation issue)
                import torch
                if device.type == 'cpu' and torch.cuda.is_available():
                    logging.warning("[Train Eval] Model is on CPU! Moving to GPU...")
                    device = torch.device("cuda")
                    model_to_eval = model_to_eval.to(device)
                    
                criterion = torch.nn.CrossEntropyLoss().to(device)
                
                total_correct = 0
                total_samples = 0
                total_loss = 0.0
                
                train_start = time.time()
                
                with torch.no_grad():
                    for batch_idx, (x, labels) in enumerate(self.train_global):
                        x, labels = x.to(device), labels.to(device)
                        outputs = model_to_eval(x)
                        loss = criterion(outputs, labels)
                        
                        _, predicted = torch.max(outputs, 1)
                        total_correct += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                        total_loss += loss.item() * labels.size(0)
                
                train_elapsed = time.time() - train_start
                train_metrics["num_samples"] = total_samples
                train_metrics["num_correct"] = total_correct
                train_metrics["losses"] = total_loss / total_samples if total_samples > 0 else 0
                
                logging.debug(f"[Train Eval] Subnet {subnet_id}: {total_samples} samples, {train_elapsed:.2f}s")
            if self.args.dataset == "shakespeare":
                self.client_trainer.update_local_dataset(
                    0,
                    self.train_data_local_dict[0],
                    eval_data,  # Use validation set for periodic eval
                    self.train_data_local_num_dict[0],
                )
            test_metrics = dict()
            # --- OPTIMIZED: Direct evaluation on (sampled) validation set ---
            val_start_time = time.time()
            
            target_model = self.client_trainer.test_model
            model_to_eval = target_model.get_model() if hasattr(target_model, 'get_model') else target_model
            model_to_eval.eval()
            
            device = next(model_to_eval.parameters()).device
            # Ensure model is on GPU if available
            if device.type == 'cpu' and torch.cuda.is_available():
                logging.warning("[Val Eval] Model is on CPU! Moving to GPU...")
                device = torch.device("cuda")
                model_to_eval = model_to_eval.to(device)

            criterion = torch.nn.CrossEntropyLoss().to(device)
            
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(eval_data):
                    x, labels = x.to(device), labels.to(device)
                    outputs = model_to_eval(x)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    val_loss += loss.item() * labels.size(0)
            
            val_elapsed = time.time() - val_start_time
            logging.debug(f"[Val Eval] Subnet {subnet_id}: {val_total} samples, {val_elapsed:.2f}s")
            
            if self.args.dataset == 'ptb':
                test_metrics["ppl"] = 0  # PTB not supported in this path
            else:
                test_metrics["num_samples"] = val_total
                test_metrics["num_correct"] = val_correct
            test_metrics["losses"] = val_loss / val_total if val_total > 0 else 0

            if self.args.verbose_test:
                print("test stats", test_metrics)

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is too slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            if self.args.dataset == 'ptb':
                if not self.args.skip_train_test:
                    train_loss = sum(train_metrics["losses"]) / self.args.client_num_in_total
                    train_ppl = sum(train_metrics["ppl"]) / self.args.client_num_in_total
                    avg_subnet_train_metrics["ppl"].append(train_ppl)
                    avg_subnet_train_metrics["losses"].append(train_loss)
                    avg_subnet_train_metrics["num_samples"].append(1)

                    wandb_subnet_log = dict()
                    if "d" in subnet_info:
                        wandb_subnet_log["d"] = subnet_info["d"]
                    if "e" in subnet_info:
                        e_list = subnet_info["e"]
                        # Check if the list is not empty and all elements are equal to the first one
                        if e_list and all(x == e_list[0] for x in e_list):
                            # If so, log only the single value
                            wandb_subnet_log["e"] = e_list[0]
                        else:
                            # Otherwise, log the full list as before
                            wandb_subnet_log["e"] = e_list
                    if "w_indices" in subnet_info: # Use "w_indices" to be consistent
                        wandb_subnet_log["w"] = subnet_info["w_indices"]

                    stats = {
                        "train_ppl": train_ppl,
                        "train_loss": train_loss,
                        # "subnet": subnet_info,
                    }
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/PPL": train_ppl, "round": round_idx,}
                    )
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,}
                    )
                    # logging.info(stats)
                    logging.info(f"subnet: {self._get_formatted_subnet_str(subnet_info)} -> {stats}")
                    subnet_train_acc_map[subnet_id] = stats["train_ppl"]

                # test on test dataset
                test_ppl = test_metrics["ppl"]
                test_loss = test_metrics["losses"]
                avg_subnet_test_metrics["ppl"].append(test_ppl)
                avg_subnet_test_metrics["losses"].append(test_loss)
                avg_subnet_test_metrics["num_samples"].append(1)

                wandb_subnet_log = dict()
                if "d" in subnet_info:
                    wandb_subnet_log["d"] = subnet_info["d"]
                if "e" in subnet_info:
                    e_list = subnet_info["e"]
                    # Check if the list is not empty and all elements are equal to the first one
                    if e_list and all(x == e_list[0] for x in e_list):
                        # If so, log only the single value
                        wandb_subnet_log["e"] = e_list[0]
                    else:
                        # Otherwise, log the full list as before
                        wandb_subnet_log["e"] = e_list
                if "w_indices" in subnet_info:
                    wandb_subnet_log["w"] = subnet_info["w_indices"]

                stats = {
                    "test_ppl": test_ppl,
                    "test_loss": test_loss,
                    # "subnet": subnet_info,
                }
                wandb.log(
                    {f"Test/{wandb_subnet_log}/PPL": test_ppl, "round": round_idx}
                )
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,}
                )
                # logging.info(stats)
                logging.info(f"subnet: {self._get_formatted_subnet_str(subnet_info)} -> {stats}")
                subnet_test_acc_map[subnet_id] = stats["test_ppl"]
            else:
                if not self.args.skip_train_test:
                    # test on train dataset (now uses pre-computed values from train_global eval)
                    train_acc = train_metrics["num_correct"] / train_metrics["num_samples"] if train_metrics["num_samples"] > 0 else 0
                    train_loss = train_metrics["losses"]  # Already averaged
                    avg_subnet_train_metrics["num_correct"].append(train_acc)
                    avg_subnet_train_metrics["losses"].append(train_loss)
                    avg_subnet_train_metrics["num_samples"].append(1)

                    wandb_subnet_log = dict()
                    if "d" in subnet_info:
                        wandb_subnet_log["d"] = subnet_info["d"]
                    if "e" in subnet_info:
                        e_list = subnet_info["e"]
                        # Check if the list is not empty and all elements are equal to the first one
                        if e_list and all(x == e_list[0] for x in e_list):
                            # If so, log only the single value
                            wandb_subnet_log["e"] = e_list[0]
                        else:
                            # Otherwise, log the full list as before
                            wandb_subnet_log["e"] = e_list
                    if "w_indices" in subnet_info:
                        wandb_subnet_log["w"] = subnet_info["w_indices"]

                    stats = {
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        # "subnet": subnet_info,
                    }
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Acc": train_acc, "round": round_idx,}
                    )
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,}
                    )
                    # logging.info(stats)
                    logging.info(f"subnet: {self._get_formatted_subnet_str(subnet_info)} -> {stats}")
                    subnet_train_acc_map[subnet_id] = stats["train_acc"]

                # test on test dataset
                test_acc = test_metrics["num_correct"] / test_metrics["num_samples"]
                test_loss = test_metrics["losses"]  # Already averaged in direct eval loop
                avg_subnet_test_metrics["num_correct"].append(test_acc)
                avg_subnet_test_metrics["losses"].append(test_loss)
                avg_subnet_test_metrics["num_samples"].append(1)

                wandb_subnet_log = dict()
                if "d" in subnet_info:
                    wandb_subnet_log["d"] = subnet_info["d"]
                if "e" in subnet_info:
                    e_list = subnet_info["e"]
                    # Check if the list is not empty and all elements are equal to the first one
                    if e_list and all(x == e_list[0] for x in e_list):
                        # If so, log only the single value
                        wandb_subnet_log["e"] = e_list[0]
                    else:
                        # Otherwise, log the full list as before
                        wandb_subnet_log["e"] = e_list
                if "w_indices" in subnet_info:
                    wandb_subnet_log["w"] = subnet_info["w_indices"]

                stats = {
                    f"{eval_label}_acc": test_acc,
                    f"{eval_label}_loss": test_loss,
                    # "subnet": subnet_info,
                }
                wandb.log(
                    {f"{eval_label.title()}/{wandb_subnet_log}/Acc": test_acc, "round": round_idx}
                )
                wandb.log(
                    {f"{eval_label.title()}/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,}
                )
                # logging.info(stats)
                logging.info(f"subnet: {self._get_formatted_subnet_str(subnet_info)} -> {stats}")
                subnet_test_acc_map[subnet_id] = stats[f"{eval_label}_acc"]

        if self.args.dataset == 'ptb':
            if not self.args.skip_train_test:
                # test on train dataset

                final_train_ppl = sum(avg_subnet_train_metrics["ppl"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )
                final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )

                final_stats = {
                    "final_train_ppl": final_train_ppl,
                    "final_train_loss": final_train_loss,
                }
                wandb.log({f"Train/PPL": final_train_ppl, "round": round_idx})
                wandb.log({f"Train/Loss": final_train_loss, "round": round_idx})
                logging.info(final_stats)

            # test on test dataset
            final_test_ppl = sum(avg_subnet_test_metrics["ppl"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )
            final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )

            final_stats = {
                "final_test_ppl": final_test_ppl,
                "final_test_loss": final_test_loss,
            }
            wandb.log({f"Test/PPL": final_test_ppl, "round": round_idx})
            wandb.log({f"Test/Loss": final_test_loss, "round": round_idx})
            logging.info(final_stats)
        else:
            if not self.args.skip_train_test:
                # test on train dataset
                final_train_acc = sum(avg_subnet_train_metrics["num_correct"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )
                final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )

                final_stats = {
                    "final_train_acc": final_train_acc,
                    "final_train_loss": final_train_loss,
                }
                wandb.log({f"Train/Acc": final_train_acc, "round": round_idx})
                wandb.log({f"Train/Loss": final_train_loss, "round": round_idx})
                logging.info(final_stats)

            # evaluate on validation/test dataset
            final_test_acc = sum(avg_subnet_test_metrics["num_correct"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )
            final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )

            final_stats = {
                f"final_{eval_label}_acc": final_test_acc,
                f"final_{eval_label}_loss": final_test_loss,
            }
            wandb.log({f"{eval_label.title()}/Acc": final_test_acc, "round": round_idx})
            wandb.log({f"{eval_label.title()}/Loss": final_test_loss, "round": round_idx})
            logging.info(final_stats)
        eval_elapsed = time.time() - eval_start_time
        # Compact per-subnet accuracy summary
        if subnet_test_acc_map:
            acc_parts = [f"s{k}={v:.4f}" for k, v in subnet_test_acc_map.items()]
            logging.info(f"--- Eval done ({eval_elapsed:.1f}s) | {' | '.join(acc_parts)} ---")
        return subnet_train_acc_map, subnet_test_acc_map

    def uniform_avg(self, round_num=None):
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(1)
        return subnet_flofa_avg_weights

    def maxnet_linear_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.weight_increment = (
                self.final_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps
            self.wt_avg_init = True
        round_num = min(round_num, self.num_steps)
        maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment * round_num
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.largest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def minnet_linear_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_minnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_minnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.weight_increment = (
                self.final_minnet_avg_wt - self.init_minnet_avg_wt
            ) / self.num_steps
            self.wt_avg_init = True
        round_num = min(round_num, self.num_steps)
        minnet_weight = self.init_minnet_avg_wt + self.weight_increment * round_num
        other_weight = (1 - minnet_weight) / (self.args.client_num_per_round - self.args.bottom_k_maxnet)
        minnet_weight /= self.args.bottom_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.smallest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = minnet_weight
                break
        return subnet_flofa_avg_weights

    def maxnet_cos_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.init_maxnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        maxnet_weight = self.init_maxnet_avg_wt * decayed
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.largest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def maxnet_cos_all_subnet_sandwich(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.init_maxnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        maxnet_weight = self.init_maxnet_avg_wt * decayed
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if idx == ((round_num + 1) % self.args.client_num_per_round):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def minnet_cos_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_minnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_minnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_minnet_avg_wt) / self.init_minnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        minnet_weight = self.init_minnet_avg_wt * decayed
        other_weight = (1 - minnet_weight) / (self.args.client_num_per_round - self.args.bottom_k_maxnet)
        minnet_weight /= self.args.bottom_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.smallest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = minnet_weight
                break
        return subnet_flofa_avg_weights

    def phased_maxnet_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.mid_maxnet_avg_wt = self.args.weighted_avg_schedule["mid"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps1 = self.args.weighted_avg_schedule["num_steps1"]
            self.num_steps2 = self.args.weighted_avg_schedule["num_steps2"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.mid_maxnet_avg_wt
            self.weight_increment1 = (
                self.mid_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps1
            self.weight_increment2 = (
                self.final_maxnet_avg_wt - self.mid_maxnet_avg_wt
            ) / self.num_steps2
            self.wt_avg_init = True
        if round_num < self.num_steps1:
            maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment1 * round_num
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights[idx] = maxnet_weight
                    break
        else:
            round_num = min(round_num, self.num_steps1 + self.num_steps2)
            cos_decay = 0.5 * (
                1 + np.cos(np.pi * round_num / self.num_steps1 + self.num_steps2)
            )
            decayed = (1 - self.alpha) * cos_decay + self.alpha
            maxnet_weight = self.init_maxnet_avg_wt * decayed
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights[idx] = maxnet_weight
                    break
        return subnet_flofa_avg_weights

    def multikd_phased_maxnet_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.mid_maxnet_avg_wt = self.args.weighted_avg_schedule["mid"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps1 = self.args.weighted_avg_schedule["num_steps1"]
            self.num_steps2 = self.args.weighted_avg_schedule["num_steps2"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.mid_maxnet_avg_wt
            self.weight_increment1 = (
                self.mid_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps1
            self.weight_increment2 = (
                self.final_maxnet_avg_wt - self.mid_maxnet_avg_wt
            ) / self.num_steps2
            self.wt_avg_init = True
        if round_num < self.num_steps1:
            maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment1 * round_num
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
                subnet_flofa_avg_weights.append(maxnet_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights.pop(idx * 2)
                    break
        else:
            round_num = min(round_num, self.num_steps1 + self.num_steps2)
            cos_decay = 0.5 * (
                1 + np.cos(np.pi * round_num / self.num_steps1 + self.num_steps2)
            )
            decayed = (1 - self.alpha) * cos_decay + self.alpha
            maxnet_weight = self.init_maxnet_avg_wt * decayed
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
                subnet_flofa_avg_weights.append(maxnet_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights.pop(idx * 2)
                    break
        return subnet_flofa_avg_weights

    # FedDyn update server state variables
    def update_server_state(self, client_updates):
        model_delta = self.server_model.get_model_copy()
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)
        global_model = self.server_model.get_model_copy()
        for client_model in client_updates:
            for server_param, client_param, delta_param in zip(
                global_model.parameters(),
                client_model.parameters(),
                model_delta.parameters(),
            ):
                delta_param.data += (
                    client_param - server_param
                ) / self.args.client_num_in_total

        for state_param, delta_param in zip(
            self.server_model.get_server_state().parameters(), model_delta.parameters(),
        ):
            state_param.data -= self.feddyn_alpha * delta_param
