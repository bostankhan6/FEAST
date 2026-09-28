import numpy as np
import copy
import random
import torch

from feast.Server.base_server_model import BaseServerModel
from feast.Client.client_model import ClientModel
from feast.elastic_nn.generic_ofa_network import GenericOFAResNet
from feast.utils.subnet_cost import subnet_macs
import wandb
import os
import logging

class GenericServerOFA(BaseServerModel):
    def __init__(
        self,
        arch_params,
        sampling_method,
        num_cli_total,
        bn_gamma_zero_init=False,
        cli_subnet_track=None,
        client_mac_budgets=None,  # NEW: Accept budgets from checkpoint
    ):
        self.arch_params = arch_params
        super(GenericServerOFA, self).__init__(
            init_params=arch_params,
            sampling_method=sampling_method,
            num_cli_total=num_cli_total,
            cli_subnet_track=cli_subnet_track,
        )
        self.num_cli_total = num_cli_total
        min_arch_config = self.min_subnet_arch()
        max_arch_config = self.max_subnet_arch()

        self._macs_min, _ = subnet_macs(
            depth_vec=min_arch_config['d'],
            exp_vec=min_arch_config['e'],
            w_indices=min_arch_config['w_indices'],
            width_mult_options=self.arch_params['width_multiplier_choices'],
            arch_config_params=self.arch_params
        )
        
        self._macs_max, _ = subnet_macs(
            depth_vec=max_arch_config['d'],
            exp_vec=max_arch_config['e'],
            w_indices=max_arch_config['w_indices'],
            width_mult_options=self.arch_params['width_multiplier_choices'],
            arch_config_params=self.arch_params
        )
        logging.info(f"GenericServerOFA: MACs range [{self._macs_min/1e6:.2f}M, {self._macs_max/1e6:.2f}M]")

        # --- Resource Heterogeneity: Assign or Restore Client Budgets ---
        if client_mac_budgets is not None:
            # Budgets provided externally (checkpoint restore or pre-computed correlated partition).
            # assign_client_resources() is skipped since budgets are already known, but we still
            # need to eagerly load the subnet cache — it is NOT persisted in checkpoints and is
            # required by get_client_local_max_info(), create_sub_supernet(), per_step_data, etc.
            self.client_mac_budgets = client_mac_budgets
            logging.info(f"Restored client_mac_budgets ({len(client_mac_budgets)} clients).")
            cache_path = self.arch_params.get('subnet_cache_path')
            if cache_path and self.subnet_cache is None:
                try:
                    self._prepare_subnet_cache({"subnet_cache_path": cache_path})
                except Exception as e:
                    print(f"Warning: Failed to load subnet cache: {e}")
            # Re-clip budgets to recalculated cache min (CSV MACs and runtime MACs
            # can differ slightly due to rounding in make_divisible)
            if hasattr(self, 'subnet_cache_macs') and self.subnet_cache_macs:
                cache_min = self.subnet_cache_macs[0]
                clipped = 0
                for cid in self.client_mac_budgets:
                    if self.client_mac_budgets[cid] < cache_min:
                        self.client_mac_budgets[cid] = cache_min
                        clipped += 1
                if clipped > 0:
                    logging.info(f"Re-clipped {clipped} client budgets to cache min ({cache_min/1e6:.2f}M).")
        else:
            # Assign new budgets based on config
            self.client_mac_budgets = {}
            self.assign_client_resources()

    def assign_client_resources(self):
        """
        Assigns fixed MAC budgets to each client based on the configured distribution.
        """
        is_hetero = self.arch_params.get('resource_heterogeneity', False)
        if not is_hetero:
            print("Resource Heterogeneity: DISABLED. All clients have unlimited budget.")
            # Explicitly set to infinity just to be safe, though usage logic handles checks
            for i in range(self.num_cli_total):
                self.client_mac_budgets[i] = float('inf')
            # Still load the subnet cache so sandwich training can look up MACs correctly
            cache_path = self.arch_params.get('subnet_cache_path')
            if cache_path and self.subnet_cache is None:
                try:
                    self._prepare_subnet_cache({"subnet_cache_path": cache_path})
                except Exception as e:
                    print(f"Warning: Failed to load subnet cache: {e}")
            return

        print("Resource Heterogeneity: ENABLED.")
        dist_type = self.arch_params.get('resource_distribution_type', 'zipf')
        alpha = self.arch_params.get('resource_zipf_alpha', 1.2)
        
        # Prepare cache to determine bounds based on actual available subnets
        cache_path = self.arch_params.get('subnet_cache_path')
        if cache_path and self.subnet_cache is None:
            # We pass a dict as args because _prepare_subnet_cache expects an object with .get()
            # or we ensure it works. BaseServerModel._prepare_subnet_cache uses args.get("subnet_cache_path").
            try:
                self._prepare_subnet_cache({"subnet_cache_path": cache_path})
            except Exception as e:
                print(f"Warning: Failed to load subnet cache for resource bounds: {e}")
        
        # Determine bounds
        # Default lower bound: Min of Subnet Cache (approx 458M) if available, else Global Min
        min_mac = self.arch_params.get('resource_min_mac')
        if min_mac is None: 
            if hasattr(self, 'subnet_cache_macs') and self.subnet_cache_macs:
                min_mac = self.subnet_cache_macs[0]
            else:
                min_mac = self._macs_min
        
        # Default upper bound: Max of Subnet Cache if available, else Global Max
        max_mac = self.arch_params.get('resource_max_mac')
        if max_mac is None:
            max_cached = 0
            if hasattr(self, 'subnet_cache_macs') and self.subnet_cache_macs:
                max_cached = self.subnet_cache_macs[-1]
            
            # Use the larger of cached max or runtime max to ensure budget sufficiency
            # This handles cases where runtime calc (updated code) > cached calc (old csv)
            max_mac = max(max_cached, self._macs_max)
 
        print(f"Resource Constraints: Type={dist_type}, Alpha={alpha}, Min={min_mac/1e6:.2f}M, Max={max_mac/1e6:.2f}M")

        if dist_type == 'uniform':
            # Uniformly distributed between min and max
            budgets = np.random.uniform(min_mac, max_mac, self.num_cli_total)
        
        elif dist_type == 'zipf':
            # Implements the paper's Zipf rank-to-budget mapping (Supplementary
            # Eq. "zipf" + Eq. "rank_to_budget"): draw a resource rank from
            # Pr[rank=r] = r^-alpha / sum_{l=1..N} l^-alpha, then map rank r to a
            # jittered budget in [min_mac, max_mac] -- rank 1 (most probable) gets
            # the lowest budget, rank N (rarest) gets the highest.
            s = np.random.zipf(alpha, self.num_cli_total)
            max_zipf_val = 1000
            s = np.clip(s, 1, max_zipf_val)

            ranks = np.arange(1, self.num_cli_total + 1)
            probs = 1.0 / np.power(ranks, alpha)
            probs /= probs.sum()
            level_indices = np.random.choice(ranks, size=self.num_cli_total, p=probs)

            # rank 1 -> min_mac, rank N -> max_mac, with uniform jitter within each step
            # so budgets are continuous rather than landing on N discrete levels.
            steps = (max_mac - min_mac) / max(1, (self.num_cli_total - 1))
            budgets = min_mac + (level_indices - 1) * steps
            jitter = np.random.uniform(-steps/2, steps/2, size=self.num_cli_total)
            budgets += jitter
        
        else:
            raise ValueError(f"Unknown resource distribution type: {dist_type}")

        # Final Clip and Assign
        budgets = np.clip(budgets, min_mac, max_mac)
        
        # Override for Guaranteed Max Tier Clients
        n_force_max = self.arch_params.get('resource_force_max_clients', 0)
        if n_force_max > 0:
            print(f"Forcing {n_force_max} clients to have Max Budget ({max_mac/1e6:.2f}M).")
            # We overwrite the first n clients. Since we shuffle immediately after, 
            # these max-budget clients will be randomly distributed.
            for i in range(min(n_force_max, self.num_cli_total)):
                budgets[i] = max_mac
        
        # Shuffle assignment so client ID 0 isn't always the same type
        np.random.shuffle(budgets)
        
        for i in range(self.num_cli_total):
            self.client_mac_budgets[i] = budgets[i]
        
        # Log distribution statistics
        b_vals = list(self.client_mac_budgets.values())
        print(f"Assigned Budgets: Mean={np.mean(b_vals)/1e6:.2f}M, Min={np.min(b_vals)/1e6:.2f}M, Max={np.max(b_vals)/1e6:.2f}M")
        
        # Save budgets to file
        try:
            import csv
            budget_file = os.path.join(wandb.run.dir, "client_budgets.csv")
            with open(budget_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Client ID", "Budget (M)"])
                for cid, budget in self.client_mac_budgets.items():
                    # Format as Millions with 2 decimal points
                    b_val = float(budget) / 1e6
                    writer.writerow([cid, f"{b_val:.2f}"])
            print(f"Client budgets saved to: {budget_file}")
            wandb.save(budget_file, base_path=wandb.run.dir)
        except Exception as e:
            print(f"Failed to save client budgets: {e}")

        # Pre-compute Coverage Map for Specific Random Sampling
        self.subnet_coverage_map = {}
        if hasattr(self, 'subnet_cache_macs') and self.subnet_cache_macs:
            n_total = self.num_cli_total
            budgets = list(self.client_mac_budgets.values())
            for idx, mac in enumerate(self.subnet_cache_macs):
                n_can_train = sum(1 for b in budgets if b >= mac)
                self.subnet_coverage_map[idx] = n_can_train / n_total

    def compute_subnet_coverage(self, local_max_mac=None):
        """
        Compute what fraction of clients can train a specific subnet tier.
        Used for adaptive KD ratio calculation.
        
        Args:
            local_max_mac: MAC of the client's local max subnet. If None, uses global max.
        
        Returns:
            coverage_stats dict with coverage_min, coverage_max (for local max), coverage_rand
        """
        n_total = len(self.client_mac_budgets)
        if n_total == 0:
            return {'coverage_min': 1.0, 'coverage_max': 1.0, 'coverage_rand': 1.0}
        
        budgets = list(self.client_mac_budgets.values())
        
        # Use local max MAC if provided, else use global max (backward compatible)
        if local_max_mac is not None:
            max_mac = local_max_mac
        else:
            # Handle case where subnet_cache_macs doesn't exist (e.g., SuperFedNAS without cache)
            if hasattr(self, 'subnet_cache_macs') and self.subnet_cache_macs:
                max_mac = self.subnet_cache_macs[-1]
            else:
                max_mac = self._macs_max
        
        # Coverage for this specific local max
        n_can_train_max = sum(1 for b in budgets if b >= max_mac)
        c_local_max = n_can_train_max / n_total
        
        # Expected random subnet coverage (Option 3: weighted by sampling probability)
        # For each subnet in cache (excluding min), compute its coverage
        if hasattr(self, 'subnet_cache_macs') and self.subnet_cache_macs and len(self.subnet_cache_macs) > 2:
            # Middle subnets (excluding min at [0] and max at [-1])
            middle_macs = self.subnet_cache_macs[1:-1]
            coverages = [sum(1 for b in budgets if b >= mac) / n_total for mac in middle_macs]
            c_rand = sum(coverages) / len(coverages) if coverages else c_local_max
        else:
            c_rand = c_local_max  # Fallback if only min/max exist or no cache
        
        coverage_stats = {
            'coverage_min': 1.0,
            'coverage_max': c_local_max,  # Now per-client based on local_max_mac
            'coverage_rand': c_rand,
            'n_clients_max': n_can_train_max,
            'n_clients_total': n_total,
            'local_max_mac': max_mac,
        }
        
        # Logging removed - info is now consolidated in client training log
        return coverage_stats


    def init_model(self, init_params):
        # Filter out keys that are not meant for the model architecture
        model_params = copy.deepcopy(init_params)
        keys_to_remove = [
            'resource_heterogeneity',
            'resource_distribution_type',
            'resource_zipf_alpha',
            'resource_min_mac',
            'resource_max_mac',
            'resource_force_max_clients',
            'subnet_cache_path',
            'diverse_subnets',
            'alpha_weights',       # Used for loss weighting, not model architecture
            'beta_depth_penalty',  # Used for cache generation, not model architecture
        ]
        for k in keys_to_remove:
            if k in model_params:
                del model_params[k]
                
        return GenericOFAResNet(**model_params)

    # --- Generic Architecture Definition & Sampling Methods ---

    def max_subnet_arch(self):
        num_stages = self.arch_params['num_stages']
        max_extra_blocks = self.arch_params['max_extra_blocks_per_stage']
        d = [max_extra_blocks] * num_stages
        max_exp_val = max(self.arch_params['expansion_ratio_choices'])
        e = [max_exp_val] * num_stages  # one per stage
        max_w_idx = len(self.arch_params['width_multiplier_choices']) - 1
        w_indices = [max_w_idx] * (num_stages + 1)
        return {"d": d, "e": e, "w_indices": w_indices}

    def min_subnet_arch(self):
        num_stages = self.arch_params['num_stages']
        d = [0] * num_stages
        min_exp_val = min(self.arch_params['expansion_ratio_choices'])
        e = [min_exp_val] * num_stages  # one per stage
        w_indices = [0] * (num_stages + 1)
        return {"d": d, "e": e, "w_indices": w_indices}

    def random_subnet_arch(self):
        num_stages = self.arch_params['num_stages']
        max_extra_blocks = self.arch_params['max_extra_blocks_per_stage']
        d = [random.randint(0, max_extra_blocks) for _ in range(num_stages)]
        e = [random.choice(self.arch_params['expansion_ratio_choices']) for _ in range(num_stages)]  # one per stage
        num_w_choices = len(self.arch_params['width_multiplier_choices'])
        w_indices = [random.randint(0, num_w_choices - 1) for _ in range(num_stages + 1)]
        return {"d": d, "e": e, "w_indices": w_indices}

    def constrained_random_subnet_arch(self, budget, max_attempts=50):
        """
        Sample a random subnet from the FULL search space that fits within the budget.
        If no valid subnet is found after max_attempts, falls back to min subnet.
        
        Args:
            budget: Maximum MACs allowed for the subnet
            max_attempts: Maximum number of sampling attempts before fallback
            
        Returns:
            arch_config dict with d, e, w_indices
        """
        from feast.utils.subnet_cost import subnet_macs
        
        for _ in range(max_attempts):
            arch_config = self.random_subnet_arch()
            
            # Calculate MACs for this random subnet
            macs, _ = subnet_macs(
                depth_vec=arch_config['d'],
                exp_vec=arch_config['e'],
                w_indices=arch_config['w_indices'],
                width_mult_options=self.arch_params['width_multiplier_choices'],
                arch_config_params=self.arch_params
            )
            
            if macs <= budget:
                return arch_config
        
        # Fallback to min subnet if no valid sample found
        logging.warning(f"[Constrained Random] No valid subnet found within {budget/1e6:.1f}M after {max_attempts} attempts. Falling back to min subnet.")
        return self.min_subnet_arch()

    def random_depth_subnet_arch(self):
        """Samples a random depth but max width/expansion."""
        arch_config = self.max_subnet_arch() # Start with the max config
        # Only randomize the depth
        num_stages = self.arch_params['num_stages']
        max_extra_blocks = self.arch_params['max_extra_blocks_per_stage']
        arch_config['d'] = [random.randint(0, max_extra_blocks) for _ in range(num_stages)]
        return arch_config
        
    def random_compound_subnet_arch(self):
        """Samples depth and width/expansion jointly. Aliased to random_subnet_arch,
        which already samples all three dimensions independently at random."""
        return self.random_subnet_arch()

    def mutate_sample(self, sample_arch, mut_prob):
        new_sample = copy.deepcopy(sample_arch)
        num_stages = self.arch_params['num_stages']
        max_extra_blocks = self.arch_params['max_extra_blocks_per_stage']
        num_w_choices = len(self.arch_params['width_multiplier_choices'])
        gene_type = random.choice(['d', 'e', 'w'])
        if gene_type == 'd':
            idx_to_mutate = random.randint(0, num_stages - 1)
            new_sample["d"][idx_to_mutate] = random.randint(0, max_extra_blocks)
        elif gene_type == 'e':
            idx_to_mutate = random.randint(0, self.arch_params['num_stages'] - 1)
            new_sample["e"][idx_to_mutate] = random.choice(self.arch_params['expansion_ratio_choices'])
        else: # 'w'
            idx_to_mutate = random.randint(0, len(new_sample["w_indices"]) - 1)
            new_sample["w_indices"][idx_to_mutate] = random.randint(0, num_w_choices - 1)
        return new_sample

    def is_max_net(self, arch):
        return arch == self.max_subnet_arch()

    def is_min_net(self, arch):
        return arch == self.min_subnet_arch()

    def arch_to_subnet_kwargs(self, arch):
        """
        Converts an architecture dictionary (with 'd', 'e', 'w_indices') 
        to the kwargs expected by GenericOFAResNet.set_active_subnet (needs 'e_indices').
        """
        d = arch['d']
        e = arch['e']
        w_indices = arch['w_indices']
        
        exp_choices = self.arch_params['expansion_ratio_choices']
        try:
            # Exact match is expected; both branches resolve identically today,
            # kept separate for a future tolerant-match fallback if needed.
            e_indices = []
            for val in e:
                if val in exp_choices:
                    e_indices.append(exp_choices.index(val))
                else:
                    e_indices.append(exp_choices.index(val))
        except ValueError as err:
            raise ValueError(f"Expansion value error in arch_to_subnet_kwargs: {err}. Value not in choices {exp_choices}.") from err
            
        return {"d": d, "e_indices": e_indices, "w_indices": w_indices}

    # --- Core Subnet Management and Aggregation ---

    def get_subnet(self, d, e, w_indices, preserve_weight=True, **kwargs):
        exp_choices = self.arch_params['expansion_ratio_choices']
        try:
            e_indices = [exp_choices.index(val) for val in e]
        except ValueError as err:
            raise ValueError(f"Expansion value error: {err}. Value not in choices {exp_choices}.") from err
        self.model.set_active_subnet(d=d, e_indices=e_indices, w_indices=w_indices)
        subnet = self.model.get_active_subnet(preserve_weight=preserve_weight)
        subindex = self.active_subnet_index()
        arch_config = {"d": d, "e": e, "w_indices": w_indices}
        new_model = ClientModel(
            subnet,
            subindex,
            arch_config,
            self.is_max_net,
            sample_random_subnet=self.random_subnet_arch,
            sample_random_depth_subnet=self.random_depth_subnet_arch,
        )
        self.model.set_max_net()
        return new_model

    def active_subnet_index(self):
        """
        Maps the currently active subnet's local block/channel indices back to
        their position in the full supernet -- this is the mapping Omega_i used
        by add_subnet()'s sparse aggregation (Supplementary Sec. "Sub-Supernet
        Construction, Sparse Aggregation, and Communication Analysis").
        """
        supernet = self.model
        mapping = {"input_stem": {}, "blocks": {}, "classifier": {}, "channels": {}}
        mapping["input_stem"][0] = 0
        mapping["channels"]["stem"] = (supernet.input_stem[0].active_out_channel, self.arch_params['initial_input_channels'])
        subnet_block_idx = 0
        all_supernet_stage_indices = supernet.grouped_block_index
        active_in_channel = supernet.input_stem[0].active_out_channel
        for stage_id in range(supernet.num_stages):
            num_active_blocks = (supernet.max_extra_blocks_per_stage + 1) - supernet.runtime_depth[stage_id]
            supernet_indices_for_active_blocks = all_supernet_stage_indices[stage_id][:num_active_blocks]
            for i, supernet_block_idx in enumerate(supernet_indices_for_active_blocks):
                supernet_block = supernet.blocks[supernet_block_idx]
                mapping["blocks"][subnet_block_idx] = supernet_block_idx
                mapping["channels"][subnet_block_idx] = (supernet_block.active_out_channel, supernet_block.active_middle_channels, active_in_channel if i == 0 else supernet_block.active_out_channel)
                subnet_block_idx += 1
            if num_active_blocks > 0:
                 active_in_channel = supernet.blocks[supernet_indices_for_active_blocks[-1]].active_out_channel
        mapping["classifier"][0] = 0
        mapping["channels"]["classifier"] = (supernet.classifier.out_features, supernet.classifier.active_in_features)
        return mapping

    def get_client_local_max_arch(self, client_idx):
        """
        Returns the architecture of the 'Local Max' subnet for a given client,
        defined as the largest subnet in the cache that fits within the client's MAC budget.
        """
        budget = self.client_mac_budgets.get(client_idx, float('inf'))

        # Without a cache there's no bounded search space to query MACs against;
        # return the (uncapped) global max and let budget checks elsewhere apply.
        if self.subnet_cache is None or not hasattr(self, 'subnet_cache_macs'):
            return self.max_subnet_arch()

        # subnet_cache_macs is the parallel MAC list _prepare_subnet_cache builds
        # alongside subnet_cache (sorted ascending), but scanned in full here rather
        # than assuming that order, so this stays correct even if it isn't.
        valid_indices = [i for i, mac in enumerate(self.subnet_cache_macs) if mac <= budget]
        if not valid_indices:
             # Budget too low for ANY cached subnet. Fallback to Global Min.
             return self.min_subnet_arch()
             
        # Pick the one with highest MACs
        best_idx = max(valid_indices, key=lambda i: self.subnet_cache_macs[i])
        return self.subnet_cache[best_idx]

    def get_client_local_max_info(self, client_idx):
        """Returns (arch, macs) of the largest subnet in cache within client budget."""
        budget = self.client_mac_budgets.get(client_idx, float('inf'))
        
        if self.subnet_cache is None or not hasattr(self, 'subnet_cache_macs'):
            # Fallback (legacy, no MACs known easily without calc)
            # Assuming legacy flow not primary for this feature.
            return self.max_subnet_arch(), 0.0
            
        valid_indices = [i for i, mac in enumerate(self.subnet_cache_macs) if mac <= budget]
        if not valid_indices:
             return self.get_min_cached_subnet_info()
             
        best_idx = max(valid_indices, key=lambda i: self.subnet_cache_macs[i])
        return self.subnet_cache[best_idx], self.subnet_cache_macs[best_idx]

    def min_cached_subnet_arch(self):
        """Returns the architecture of the smallest subnet in the cache."""
        arch, _ = self.get_min_cached_subnet_info()
        return arch

    def get_min_cached_subnet_info(self):
        """Returns (arch, macs) of the smallest subnet in the cache."""
        if self.subnet_cache is not None and len(self.subnet_cache) > 0:
            min_idx = min(range(len(self.subnet_cache_macs)), key=lambda i: self.subnet_cache_macs[i])
            return self.subnet_cache[min_idx], self.subnet_cache_macs[min_idx]
        return self.min_subnet_arch(), 0.0

    def random_strictly_constrained_cached_subnet_arch(self, min_macs, max_macs):
        """
        Samples a random architecture from the subnet_cache strictly BETWEEN min_macs and max_macs.
        Returns (arch, index) tuple. Returns (None, None) if no such subnet exists.
        """
        if self.subnet_cache is not None and len(self.subnet_cache) > 0:
            # Strictly between: min_macs < m < max_macs
            # Use small epsilon if float comparison issues arise, but usually < is fine.
            valid_indices = [i for i, m in enumerate(self.subnet_cache_macs) if min_macs < m < max_macs]
            
            if not valid_indices:
                 return None, None
            
            random_idx = random.choice(valid_indices)
            return self.subnet_cache[random_idx], random_idx
        return None, None

    def random_constrained_cached_subnet_arch(self, budget_macs):
        """Samples a random architecture from the subnet_cache that fits within the budget."""
        if self.subnet_cache is not None and len(self.subnet_cache) > 0:
            valid_indices = [i for i, m in enumerate(self.subnet_cache_macs) if m <= budget_macs]
            if not valid_indices:
                 # Should not happen as Min is always very small, but fallback to absolute Min
                 return self.min_subnet_arch()
            
            random_idx = random.choice(valid_indices)
            return self.subnet_cache[random_idx]
        else:
            return self.random_subnet_arch()

    def random_cached_subnet_arch(self):
        """Samples a random architecture from the subnet_cache."""
        if self.subnet_cache is not None and len(self.subnet_cache) > 0:
            random_idx = random.randint(0, len(self.subnet_cache) - 1)
            # The cache keys are string representations, so we need to access values or indexable list
            # BaseServerModel._prepare_subnet_cache stores in self.subnet_cache (list of dicts)
            return self.subnet_cache[random_idx]
        else:
            # Fallback if cache not loaded
            return self.random_subnet_arch()

    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        # Handle cases where avg_weight might not be set (e.g. raw Supernet)
        weight = getattr(w_local, 'avg_weight', 1.0) 
        local_params = w_local.state_dict()
        
        # Check if w_local is a Supernet (GenericOFAResNet) directly
        # If so, keys map 1-to-1 to the Global Supernet.
        if isinstance(w_local, GenericOFAResNet):
            for key in local_params:
                if "num_batches_tracked" in key:
                    continue
                if key in shared_param_sum:
                    # Direct aggregation
                    shared_param_sum[key] += weight * local_params[key]
                    shared_param_count[key] += weight

                else:
                    logging.warning(f"Warning: Supernet key {key} not found in global accumulation dict.")
            return

        # Standard Logic: w_local is ClientModel (Subnet) with sparse mapping
        local_index = w_local.model_index

        for key in local_params:
            if "num_batches_tracked" in key:
                continue

            supernet_key = key
            split_key = key.split('.')
            if len(split_key) > 1 and split_key[1].isdigit():
                layer_type, local_idx_str = split_key[0], split_key[1]
                if layer_type in local_index:
                    supernet_idx = local_index[layer_type].get(int(local_idx_str))
                    if supernet_idx is not None:
                        split_key[1] = str(supernet_idx)
                        supernet_key = ".".join(split_key)

            # FIX: Added key name translation for conv, bn, and linear layers
            # This ensures keys from the static subnet match the dynamic supernet's state_dict keys.
            if ".linear." in supernet_key:
                supernet_key = supernet_key.replace(".linear.", ".linear.linear.")
            if "bn." in supernet_key:
                supernet_key = supernet_key.replace("bn.", "bn.bn.")
            if "conv.weight" in supernet_key:
                supernet_key = supernet_key.replace("conv.weight", "conv.conv.weight")
            
            if supernet_key not in shared_param_sum:
                continue

            if "conv.weight" in key:
                local_idx = int(key.split('.')[1])
                if "input_stem" in key:
                    out_ch, in_ch = local_index["channels"]["stem"]
                else:
                    # Unpack channels for the block: (output, middle, input)
                    out_ch, mid_ch, in_ch = local_index["channels"][local_idx]
                    # Determine the correct in/out channels for the specific convolution layer
                    if 'conv1' in key: # First convolution in block: in -> mid
                        out_ch, in_ch = mid_ch, in_ch
                    elif 'conv2' in key: # Second convolution in block: mid -> out
                        out_ch, in_ch = out_ch, mid_ch
                    elif 'downsample' in key: # Downsample convolution: in -> out
                        # out_ch and in_ch are already correct from the tuple unpacking
                        pass
                shared_param_sum[supernet_key][:out_ch, :in_ch, :, :] += weight * local_params[key]
                shared_param_count[supernet_key][:out_ch, :in_ch, :, :] += weight
            elif "linear.weight" in key:
                out_feat, in_feat = local_index["channels"]["classifier"]
                shared_param_sum[supernet_key][:out_feat, :in_feat] += weight * local_params[key]
                shared_param_count[supernet_key][:out_feat, :in_feat] += weight
            else: # Handles biases and BN parameters
                if len(local_params[key].shape) > 0:
                    active_dim = local_params[key].shape[0]
                    shared_param_sum[supernet_key][:active_dim] += weight * local_params[key]
                    shared_param_count[supernet_key][:active_dim] += weight
                else: # For scalar values like num_batches_tracked (already skipped, but as a safeguard)
                    shared_param_sum[supernet_key] += weight * local_params[key]
                    shared_param_count[supernet_key] += weight

    # --- Sub-Supernet Methods for Resource-Heterogeneous Communication ---

    def compute_sub_supernet_bounds(self, client_budget: float) -> dict:
        """
        Analyze subnet cache to determine architecture bounds for a client.

        Args:
            client_budget: Maximum MACs allowed for the client

        Returns:
            Dictionary with:
                - max_d: [int, int, int, int] - Max extra blocks per stage
                - max_w_indices: [int, int, int, int, int] - Max width index per position
                - max_extra_blocks_per_stage: int - Overall max depth
                - width_multiplier_choices: list - Reduced width choices
                - valid_cache_indices: list - Indices of valid configs
        """
        if (self.subnet_cache is None
                or not hasattr(self, 'subnet_cache_macs')
                or not self.subnet_cache_macs):   # empty list if cache load failed mid-way
            logging.warning("compute_sub_supernet_bounds: No subnet cache available, returning full bounds")
            return self._get_full_supernet_bounds()

        # Find all configs within budget
        valid_indices = [i for i, mac in enumerate(self.subnet_cache_macs)
                         if mac <= client_budget]

        if not valid_indices:
            logging.debug(f"compute_sub_supernet_bounds: No valid subnets for budget {client_budget/1e6:.2f}M, using min bounds")
            return self._get_min_supernet_bounds()

        valid_configs = [self.subnet_cache[i] for i in valid_indices]
        num_stages = self.arch_params['num_stages']

        # Compute max across each dimension
        max_d = [max(cfg['d'][i] for cfg in valid_configs) for i in range(num_stages)]
        max_w_indices = [max(cfg['w_indices'][i] for cfg in valid_configs)
                         for i in range(num_stages + 1)]

        # Determine reduced choices
        max_w_idx_overall = max(max_w_indices)
        reduced_width_choices = self.arch_params['width_multiplier_choices'][:max_w_idx_overall + 1]

        return {
            'max_d': max_d,
            'max_w_indices': max_w_indices,
            'max_extra_blocks_per_stage': max(max_d),
            'width_multiplier_choices': reduced_width_choices,
            'valid_cache_indices': valid_indices,
            'client_budget': client_budget,
        }

    def _get_full_supernet_bounds(self) -> dict:
        """Return bounds corresponding to the full supernet."""
        num_stages = self.arch_params['num_stages']
        max_extra = self.arch_params['max_extra_blocks_per_stage']
        max_w_idx = len(self.arch_params['width_multiplier_choices']) - 1

        return {
            'max_d': [max_extra] * num_stages,
            'max_w_indices': [max_w_idx] * (num_stages + 1),
            'max_extra_blocks_per_stage': max_extra,
            'width_multiplier_choices': self.arch_params['width_multiplier_choices'],
            # Only return indices if MACs were fully computed (non-empty macs list)
            'valid_cache_indices': (list(range(len(self.subnet_cache_macs)))
                                    if (self.subnet_cache and
                                        hasattr(self, 'subnet_cache_macs') and
                                        self.subnet_cache_macs)
                                    else []),
            'client_budget': float('inf'),
        }

    def _get_min_supernet_bounds(self) -> dict:
        """Return bounds corresponding to the minimum CACHED subnet.

        Uses the actual minimum cached subnet's architecture (e.g. w_indices=[2,0,1,1,0],
        d=[5,6,7,8]) rather than the absolute global minimum ([0,0,0,0,0]).  This is
        critical because arch_bundle['min'] is always set from the cached minimum, so the
        sub-supernet must be large enough to accommodate it.  The scenario arises when a
        client's budget is just below the cache minimum (e.g. 49.58M vs 49.59M due to
        floating-point jitter) — we still assign them the minimum cached subnet.
        """
        if (self.subnet_cache and
                hasattr(self, 'subnet_cache_macs') and
                self.subnet_cache_macs):
            min_arch = self.subnet_cache[0]  # cache is sorted ascending by MACs
            max_d = list(min_arch['d'])
            max_w_indices = list(min_arch['w_indices'])
            max_w_idx_overall = max(max_w_indices)
            reduced_width_choices = self.arch_params['width_multiplier_choices'][:max_w_idx_overall + 1]
            return {
                'max_d': max_d,
                'max_w_indices': max_w_indices,
                'max_extra_blocks_per_stage': max(max_d),
                'width_multiplier_choices': reduced_width_choices,
                'valid_cache_indices': [0],
                'client_budget': self.subnet_cache_macs[0],
            }

        # Fallback when cache is unavailable (should not normally happen after __init__ fix)
        num_stages = self.arch_params['num_stages']
        return {
            'max_d': [0] * num_stages,
            'max_w_indices': [0] * (num_stages + 1),
            'max_extra_blocks_per_stage': 0,
            'width_multiplier_choices': [self.arch_params['width_multiplier_choices'][0]],
            'valid_cache_indices': [],
            'client_budget': 0,
        }

    def _compute_stage_channels(self, stage_id: int, w_index: int) -> int:
        """Compute output channels for a stage given width index."""
        from ofa.utils import make_divisible
        base_channels = self.arch_params['original_stage_base_channels'][stage_id]
        width_mult = self.arch_params['width_multiplier_choices'][w_index]
        channels = make_divisible(
            int(base_channels * width_mult),
            self.arch_params.get('channel_divisible_by', 8)
        )
        return max(channels, self.arch_params.get('channel_divisible_by', 8))

    def _compute_stem_channels(self, w_index: int) -> int:
        """Compute output channels for the stem given width index."""
        from ofa.utils import make_divisible
        base_channels = self.arch_params['original_stem_out_channels']
        width_mult = self.arch_params['width_multiplier_choices'][w_index]
        channels = make_divisible(
            int(base_channels * width_mult),
            self.arch_params.get('channel_divisible_by', 8)
        )
        return max(channels, self.arch_params.get('channel_divisible_by', 8))

    def create_sub_supernet(self, client_budget: float):
        """
        Create a smaller GenericOFAResNet for client's budget.

        Args:
            client_budget: Maximum MACs allowed for the client

        Returns:
            Tuple of (sub_supernet, sub_info) where sub_info contains:
                - mapping: Dict for aggregating back to global supernet
                - bounds: The computed bounds
                - sub_cache: Filtered subnet cache for this client
                - sub_cache_macs: MACs for filtered cache
        """
        bounds = self.compute_sub_supernet_bounds(client_budget)

        # Check if sub-supernet equals full supernet (no reduction needed).
        # "Full" means every per-stage depth is at the global max AND every
        # per-position width index is at the global max.
        full_depth = self.arch_params['max_extra_blocks_per_stage']
        global_max_w_idx = len(self.arch_params['width_multiplier_choices']) - 1
        num_stages = self.arch_params['num_stages']

        is_full_width = all(idx == global_max_w_idx for idx in bounds['max_w_indices'])
        is_full_depth = all(d == full_depth for d in bounds['max_d'])

        if is_full_width and is_full_depth:
            logging.debug(f"create_sub_supernet: Budget {client_budget/1e6:.2f}M needs full supernet")
            return self.model, {
                'mapping': None,  # None indicates full supernet (1-to-1 mapping)
                'bounds': bounds,
                'sub_cache': getattr(self, 'subnet_cache', None),
                'sub_cache_macs': getattr(self, 'subnet_cache_macs', []),
                'is_full_supernet': True,
            }

        # Build sub_arch_params with two per-position reductions:
        #
        # 1. per_position_max_w_indices — each stage only allocates channels up to its
        #    own peak width index (not the global/stem max). A 200M-budget Stage 3 no
        #    longer wastes memory on 2048 channels it can never use.
        #
        # 2. per_position_max_d — each stage only allocates block weight tensors up to
        #    its own peak depth. A shallow-budget stage 0 (max_d=0) gets 1 block slot,
        #    not 6 (which is the full supernet's global max).
        #
        # max_extra_blocks_per_stage stays at the GLOBAL value — per_position_max_d
        # limits the depth per stage; the global max defines the full supernet structure.
        sub_arch_params = copy.deepcopy(self.arch_params)
        sub_arch_params['per_position_max_w_indices'] = bounds['max_w_indices']
        sub_arch_params['per_position_max_d'] = bounds['max_d']
        # width_multiplier_choices and max_extra_blocks_per_stage stay unchanged —
        # they are the global reference values for the sub-supernet constructor.

        # Remove non-model keys
        keys_to_remove = [
            'resource_heterogeneity', 'resource_distribution_type',
            'resource_zipf_alpha', 'resource_min_mac', 'resource_max_mac',
            'resource_force_max_clients', 'subnet_cache_path', 'diverse_subnets',
            'alpha_weights',       # Used for loss weighting, not model architecture
            'beta_depth_penalty',  # Used for cache generation, not model architecture
        ]
        for k in keys_to_remove:
            sub_arch_params.pop(k, None)

        # Create sub-supernet
        sub_supernet = GenericOFAResNet(**sub_arch_params)
        sub_supernet = sub_supernet.to(next(self.model.parameters()).device)

        # Copy weights from global supernet
        mapping = self._copy_weights_to_sub_supernet(
            sub_supernet,
            bounds['max_w_indices'],
            bounds['max_d']
        )

        # Filter subnet cache for client
        sub_cache = [self.subnet_cache[i] for i in bounds['valid_cache_indices']]
        sub_cache_macs = [self.subnet_cache_macs[i] for i in bounds['valid_cache_indices']]

        logging.debug(f"Created sub-supernet: budget={client_budget/1e6:.2f}M, "
                      f"max_d={bounds['max_d']}, max_w={bounds['max_w_indices']}, "
                      f"params={sum(p.numel() for p in sub_supernet.parameters())/1e6:.2f}M")

        return sub_supernet, {
            'mapping': mapping,
            'bounds': bounds,
            'sub_cache': sub_cache,
            'sub_cache_macs': sub_cache_macs,
            'is_full_supernet': False,
        }

    def _copy_weights_to_sub_supernet(self, sub_supernet, max_w_indices, max_d) -> dict:
        """
        Copy weight slices from global supernet to sub-supernet.

        Rather than computing expected dimensions, we query the sub-supernet's state_dict
        to get actual layer shapes and copy matching slices from the global supernet.

        Args:
            sub_supernet: The smaller GenericOFAResNet to copy weights into
            max_w_indices: Max width index per position [stem, stage0, stage1, stage2, stage3]
            max_d: Max extra blocks per stage [stage0, stage1, stage2, stage3]

        Returns:
            mapping: Dict of {sub_key: (global_key, slice_info)} for aggregation
        """
        mapping = {}
        global_state = self.model.state_dict()
        sub_state = sub_supernet.state_dict()
        num_stages = self.arch_params['num_stages']

        # Build block index mapping: sub_block_idx -> global_block_idx
        global_grouped = self.model.grouped_block_index
        block_mapping = {}  # sub_block_idx -> global_block_idx
        sub_block_idx = 0
        for stage_id in range(num_stages):
            num_blocks_in_sub = max_d[stage_id] + 1
            for local_block_idx in range(num_blocks_in_sub):
                global_block_idx = global_grouped[stage_id][local_block_idx]
                block_mapping[sub_block_idx] = global_block_idx
                sub_block_idx += 1

        # Copy weights by querying sub-supernet's actual shapes
        for sub_key in sub_state:
            sub_shape = sub_state[sub_key].shape

            # Determine the global key by mapping block indices
            global_key = sub_key
            if sub_key.startswith('blocks.'):
                # Extract block index from key like "blocks.5.conv1...."
                parts = sub_key.split('.')
                sub_blk_idx = int(parts[1])
                if sub_blk_idx in block_mapping:
                    global_blk_idx = block_mapping[sub_blk_idx]
                    parts[1] = str(global_blk_idx)
                    global_key = '.'.join(parts)

            if global_key not in global_state:
                logging.warning(f"_copy_weights: Global key {global_key} not found for sub_key {sub_key}")
                continue

            global_shape = global_state[global_key].shape

            # Skip num_batches_tracked (scalar)
            if 'num_batches_tracked' in sub_key:
                sub_state[sub_key] = global_state[global_key].clone()
                mapping[sub_key] = (global_key, {'type': 'other'})
                continue

            # Copy based on tensor dimensions
            if len(sub_shape) == 4:
                # Conv weight: [out_ch, in_ch, k, k]
                out_ch, in_ch = sub_shape[0], sub_shape[1]
                sub_state[sub_key] = global_state[global_key][:out_ch, :in_ch, :, :].clone()
                mapping[sub_key] = (global_key, {'type': 'conv', 'out_ch': out_ch, 'in_ch': in_ch})

            elif len(sub_shape) == 2:
                # Linear weight: [out_feat, in_feat]
                out_feat, in_feat = sub_shape[0], sub_shape[1]
                sub_state[sub_key] = global_state[global_key][:out_feat, :in_feat].clone()
                mapping[sub_key] = (global_key, {'type': 'linear', 'out_feat': out_feat, 'in_feat': in_feat})

            elif len(sub_shape) == 1:
                # BN params or bias: [dim]
                dim = sub_shape[0]
                sub_state[sub_key] = global_state[global_key][:dim].clone()
                mapping[sub_key] = (global_key, {'type': 'bn', 'dim': dim})

            elif len(sub_shape) == 0:
                # Scalar (like num_batches_tracked)
                sub_state[sub_key] = global_state[global_key].clone()
                mapping[sub_key] = (global_key, {'type': 'other'})

            else:
                # Unknown shape - copy as-is if shapes match
                if sub_shape == global_shape:
                    sub_state[sub_key] = global_state[global_key].clone()
                    mapping[sub_key] = (global_key, {'type': 'other'})
                else:
                    logging.warning(f"_copy_weights: Shape mismatch for {sub_key}: sub={sub_shape}, global={global_shape}")

        # Load the copied state dict into sub-supernet
        sub_supernet.load_state_dict(sub_state)
        return mapping

    def add_sub_supernet(self, shared_param_sum, shared_param_count,
                          sub_state_dict, mapping, weight=1.0):
        """
        Aggregate client's sub-supernet updates into global supernet accumulators.

        Args:
            shared_param_sum: Accumulator for weighted parameter sums
            shared_param_count: Accumulator for weight counts
            sub_state_dict: State dict from client's trained sub-supernet
            mapping: {sub_key: (global_key, slice_info)} from create_sub_supernet
            weight: Aggregation weight for this client
        """
        if mapping is None:
            # Full supernet - use direct aggregation
            for key, param in sub_state_dict.items():
                if "num_batches_tracked" in key:
                    continue
                if key in shared_param_sum:
                    shared_param_sum[key] += weight * param
                    shared_param_count[key] += weight
            return

        # Sub-supernet - use sparse aggregation with mapping
        for sub_key, param in sub_state_dict.items():
            if "num_batches_tracked" in sub_key:
                continue
            if sub_key not in mapping:
                continue

            global_key, slice_info = mapping[sub_key]
            slice_type = slice_info.get('type', 'other')

            if global_key not in shared_param_sum:
                logging.warning(f"add_sub_supernet: Global key {global_key} not found in accumulators")
                continue

            if slice_type == 'conv':
                out_ch = slice_info['out_ch']
                in_ch = slice_info['in_ch']
                shared_param_sum[global_key][:out_ch, :in_ch, :, :] += weight * param
                shared_param_count[global_key][:out_ch, :in_ch, :, :] += weight

            elif slice_type == 'linear':
                out_feat = slice_info['out_feat']
                in_feat = slice_info['in_feat']
                shared_param_sum[global_key][:out_feat, :in_feat] += weight * param
                shared_param_count[global_key][:out_feat, :in_feat] += weight

            elif slice_type == 'bn' or slice_type == 'bias':
                dim = slice_info['dim']
                shared_param_sum[global_key][:dim] += weight * param
                shared_param_count[global_key][:dim] += weight

            else:  # 'other' - shapes should match
                shared_param_sum[global_key] += weight * param
                shared_param_count[global_key] += weight
