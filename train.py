import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import wandb
from collections import OrderedDict

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

# --- Argument Parsing ---
from parse_args import add_args

# --- Data Loaders ---
from feast.data.cifar100.data_loader import load_partition_data_cifar100

from feast.data.cinic10.data_loader import load_partition_data_cinic10
from feast.data.tinyimagenet.data_loader import load_partition_data_tinyimagenet

# --- Trainers ---
from feast.Server.feast_trainer import FeastTrainer
from feast.Client.subnet_trainer import SubnetTrainer
import feast
from feast.Server.generic_server_model import GenericServerOFA

def _precompute_client_budgets(args):
    """
    Pre-compute client MAC budgets using the same Zipf/uniform sampling logic as
    GenericServerOFA.assign_client_resources(). Called BEFORE load_data so the
    budgets are available for asymmetric Dirichlet data partitioning.

    Replicates the exact RNG call sequence from assign_client_resources() so that
    the result is identical to what GenericServerOFA would have computed, allowing
    GenericServerOFA to skip re-sampling when the pre-computed dict is passed in.

    Returns: dict {client_idx: float(mac_budget)}
    """
    import csv as _csv

    n_clients = args.client_num_in_total
    dist_type = args.resource_distribution_type
    alpha = args.resource_zipf_alpha
    n_force_max = args.resource_force_max_clients

    # Determine min/max from args or read from the subnet cache CSV
    min_mac = args.resource_min_mac
    max_mac = args.resource_max_mac
    if min_mac is None or max_mac is None:
        cache_macs = []
        with open(args.subnet_cache_path, 'r') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                cache_macs.append(float(row['macs']))
        if min_mac is None:
            min_mac = cache_macs[0]
        if max_mac is None:
            max_mac = cache_macs[-1]

    if dist_type == 'uniform':
        budgets = np.random.uniform(min_mac, max_mac, n_clients)
    elif dist_type == 'zipf':
        ranks = np.arange(1, n_clients + 1)
        probs = 1.0 / np.power(ranks, alpha)
        probs /= probs.sum()
        level_indices = np.random.choice(ranks, size=n_clients, p=probs)
        steps = (max_mac - min_mac) / max(1, (n_clients - 1))
        budgets = min_mac + (level_indices - 1) * steps
        jitter = np.random.uniform(-steps / 2, steps / 2, size=n_clients)
        budgets += jitter
    else:
        raise ValueError(f"Unknown resource distribution type: {dist_type}")

    budgets = np.clip(budgets, min_mac, max_mac)

    if n_force_max > 0:
        for i in range(min(n_force_max, n_clients)):
            budgets[i] = max_mac

    np.random.shuffle(budgets)

    return {i: float(budgets[i]) for i in range(n_clients)}


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        raise ValueError("Not supported")

    elif dataset_name == "femnist":
        raise ValueError("Not supported")
    #renamed from fed_shakespeare to tf_shakespeare since this is the Tensorflow sourced version of shakespeare dataset
    elif dataset_name == "tf_shakespeare":
        raise ValueError("Not supported")

    elif dataset_name == "fed_cifar100":
        raise ValueError("Not supported")
    elif dataset_name == "stackoverflow_lr":
        raise ValueError("Not supported")
    elif dataset_name == "stackoverflow_nwp":
        raise ValueError("Not supported")

    elif dataset_name == "ILSVRC2012":
        raise ValueError("Not supported")

    elif dataset_name == "gld23k":
        raise ValueError("Not supported")

    elif dataset_name == "gld160k":
        raise ValueError("Not supported")
    else:
        if dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
            (
                train_data_num,
                val_data_num,
                test_data_num,
                train_data_global,
                val_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
                bn_calibration_global,
                bn_cal_data_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.val_batch_size,
                validation_split=args.validation_split,
                bn_calibration_split=args.reset_bn_sample_size,
                client_budgets=getattr(args, '_pre_computed_budgets', None),
                corr_gamma=getattr(args, 'corr_gamma', 1.0),
                max_training_mac=getattr(args, 'max_training_mac', 600_000_000),
                augmentation=getattr(args, 'augmentation', 'basic'),
            )
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
            (
                train_data_num,
                val_data_num,
                test_data_num,
                train_data_global,
                val_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
                bn_calibration_global,
                bn_cal_data_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.val_batch_size,
                bn_calibration_split=args.reset_bn_sample_size,
                client_budgets=getattr(args, '_pre_computed_budgets', None),
                corr_gamma=getattr(args, 'corr_gamma', 1.0),
                max_training_mac=getattr(args, 'max_training_mac', 600_000_000),
                augmentation=getattr(args, 'augmentation', 'basic'),
            )
        elif dataset_name == "tinyimagenet":
            data_loader = load_partition_data_tinyimagenet
            (
                train_data_num,
                val_data_num,
                test_data_num,
                train_data_global,
                val_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
                bn_calibration_global,
                bn_cal_data_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.val_batch_size,
                validation_split=args.validation_split,
                bn_calibration_split=args.reset_bn_sample_size,
                client_budgets=getattr(args, '_pre_computed_budgets', None),
                corr_gamma=getattr(args, 'corr_gamma', 1.0),
                max_training_mac=getattr(args, 'max_training_mac', 600_000_000),
                augmentation=getattr(args, 'augmentation', 'basic'),
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    if centralized:
        train_data_local_num_dict = {
            0: sum(
                user_train_data_num
                for user_train_data_num in train_data_local_num_dict.values()
            )
        }
        train_data_local_dict = {
            0: [
                batch
                for cid in sorted(train_data_local_dict.keys())
                for batch in train_data_local_dict[cid]
            ]
        }
        test_data_local_dict = {
            0: [
                batch
                for cid in sorted(test_data_local_dict.keys())
                for batch in test_data_local_dict[cid]
            ]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid])
            for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {
            cid: combine_batches(test_data_local_dict[cid])
            for cid in test_data_local_dict.keys()
        }
        args.batch_size = args_batch_size

    # Dataset array index mapping:
    # 0: train_data_num, 1: val_data_num, 2: test_data_num
    # 3: train_data_global, 4: val_data_global, 5: test_data_global
    # 6: train_data_local_num_dict, 7: train_data_local_dict, 8: test_data_local_dict
    # 9: class_num
    # 10: bn_calibration_global (disjoint BN-recalibration set, all three datasets)
    dataset = [
        train_data_num,      # 0
        val_data_num if 'val_data_num' in dir() else 0,  # 1
        test_data_num,       # 2
        train_data_global,   # 3
        val_data_global if 'val_data_global' in dir() else None,  # 4
        test_data_global,    # 5
        train_data_local_num_dict,  # 6
        train_data_local_dict,      # 7
        test_data_local_dict,       # 8
        class_num,           # 9
        bn_calibration_global if 'bn_calibration_global' in dir() else None,  # 10: BN calibration set
    ]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, output_dim, device, load_teacher=False):
    """
    Handles creation of a new model or loading a model from a checkpoint.
    The central model factory for the generic framework.
    """
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (args.model, output_dim)
    )

    # Determine which checkpoint path to use (for main model or teacher)
    ckpt_path = args.local_model_ckpt_path if not load_teacher else None # Add specific teacher path arg if needed

    # --- Case 1: Loading from a Checkpoint ---
    if ckpt_path and os.path.exists(ckpt_path):
        logging.info(f"Loading model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Restore RNG states if resuming the main model
        if args.resume_round > 0 and not load_teacher:
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
                logging.info("Restored PyTorch RNG state from checkpoint.")
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
                logging.info("Restored NumPy RNG state from checkpoint.")
            if "random_state" in checkpoint:
                import random
                random.setstate(checkpoint["random_state"])
                logging.info("Restored Python random RNG state from checkpoint.")
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
                logging.info("Restored CUDA RNG state from checkpoint.")

        # Load the architecture parameters from the checkpoint
        if "arch_params" not in checkpoint:
            raise ValueError(f"Checkpoint '{ckpt_path}' is missing the required 'arch_params' dictionary.")
        
        loaded_arch_params = checkpoint["arch_params"]

        # Override path-based fields with CLI args (checkpoint stores absolute paths
        # from the original machine, which won't exist when resuming elsewhere)
        if args.subnet_cache_path:
            loaded_arch_params['subnet_cache_path'] = args.subnet_cache_path
            logging.info(f"Overriding subnet_cache_path from CLI: {args.subnet_cache_path}")

        # Inject classifier dropout from CLI (older checkpoints don't store this)
        classifier_dropout = getattr(args, 'classifier_dropout', 0.0)
        if classifier_dropout > 0.0:
            loaded_arch_params['dropout_rate'] = classifier_dropout
            logging.info(f"Injecting classifier_dropout={classifier_dropout} into loaded arch_params")

        logging.info(f"Loaded architecture parameters from checkpoint: {loaded_arch_params}")

        # Instantiate the model using the loaded architecture parameters
        model = GenericServerOFA(
            arch_params=loaded_arch_params,
            sampling_method=args.subnet_dist_type,
            num_cli_total=args.client_num_in_total,
            bn_gamma_zero_init=args.bn_gamma_zero_init,
            cli_subnet_track=checkpoint.get("cli_subnet_track"),  # Load tracker from checkpoint
            client_mac_budgets=checkpoint.get("client_mac_budgets"),  # Load budgets from checkpoint
        )

        # Load the model weights
        if "params" in checkpoint:
            model.set_model_params(checkpoint["params"])
            logging.info("Successfully loaded model weights from checkpoint.")
        else:
            raise ValueError(f"Checkpoint '{ckpt_path}' is missing the model weights ('params' key).")

    # --- Case 2: Creating a New Model from Scratch ---
    else:
        if ckpt_path:
            logging.warning(f"Checkpoint path specified but not found: {ckpt_path}. Creating a new model.")
        else:
            logging.info("No checkpoint path specified. Creating a new model from command-line arguments.")

        # Assemble architecture parameters from command-line arguments
        arch_params = {
            'num_stages': args.supernet_num_stages,
            'initial_input_hw': args.supernet_initial_input_hw,
            'initial_input_channels': args.supernet_initial_input_channels,
            'stem_stride': args.supernet_stem_stride,
            'original_stem_out_channels': args.supernet_original_stem_out_channels,
            'original_stage_base_channels': args.supernet_original_stage_base_channels,
            'stage_downsample_factors': args.supernet_stage_downsample_factors,
            'max_extra_blocks_per_stage': args.supernet_max_extra_blocks_per_stage,
            'channel_divisible_by': args.supernet_channel_divisible_by,
            'width_multiplier_choices': args.supernet_width_multiplier_choices,
            'expansion_ratio_choices': args.supernet_expansion_ratio_choices,
            'n_classes': output_dim,
            'bn_gamma_zero_init': args.bn_gamma_zero_init,
            'n_classes': output_dim,
            'bn_gamma_zero_init': args.bn_gamma_zero_init,
            # Resource Heterogeneity
            'resource_heterogeneity': args.resource_heterogeneity,
            'resource_distribution_type': args.resource_distribution_type,
            'resource_zipf_alpha': args.resource_zipf_alpha,
            'resource_min_mac': args.resource_min_mac,
            'resource_max_mac': args.resource_max_mac,
            'subnet_cache_path': args.subnet_cache_path,
            'resource_force_max_clients': args.resource_force_max_clients,
        }
        
        if args.model == 'ofaresnet_generic':
            model = GenericServerOFA(
                arch_params=arch_params,
                sampling_method=args.subnet_dist_type,
                num_cli_total=args.client_num_in_total,
                bn_gamma_zero_init=args.bn_gamma_zero_init,
                cli_subnet_track=args.cli_subnet_track,
                client_mac_budgets=getattr(args, '_pre_computed_budgets', None),
            )
        else:
            raise ValueError(f"Model type '{args.model}' is not supported for new model creation.")

    model.checkpoint_dir = getattr(args, "checkpoint_dir", None)
    return model


def custom_server_trainer(server_trainer_params):
    assert server_trainer_params is not None
    return FeastTrainer(**server_trainer_params)


def custom_client_trainer(client_trainer_params):
    assert client_trainer_params is not None
    return SubnetTrainer(**client_trainer_params)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone-generic"))
    args = parser.parse_args()
    logger.debug(args)

    # KD logic: inverse_kd_sandwich/feast use in-place KD (no external teacher needed),
    # so the external-teacher requirement below only applies to other strategies.
    inplace_kd_strategies = ['inverse_kd_sandwich', 'feast']
    if args.kd_ratio > 0 and not args.multi and args.training_strategy not in inplace_kd_strategies:
        assert (
            args.teacher_ckpt_name is not None and args.teacher_run_path is not None
        ), "Specify Pretrained model for knowledge distillation"

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    if getattr(args, 'multi_gpu', False):
        if not torch.cuda.is_available():
            logger.warning("[multi_gpu] --multi_gpu requested but CUDA is not available; falling back to sequential.")
            args.multi_gpu = False
        elif args.training_strategy not in inplace_kd_strategies:
            raise ValueError(
                f"--multi_gpu is currently only supported for in-place KD strategies "
                f"({inplace_kd_strategies}) where teacher_model=None. "
                f"Got '{args.training_strategy}'. Remove --multi_gpu or use "
                f"--training_strategy feast."
            )

    if getattr(args, 'multi_gpu', False):
        n_gpus = torch.cuda.device_count()
        n_workers = args.num_parallel_clients if args.num_parallel_clients is not None else n_gpus
        if n_workers < 1:
            raise ValueError(f"--num_parallel_clients must be >= 1, got {n_workers}")
        args.client_devices = [torch.device(f"cuda:{i % n_gpus}") for i in range(n_workers)]
        args.num_parallel_clients = n_workers
        logger.info(
            f"[multi_gpu] {n_workers} parallel client workers across {n_gpus} visible GPU(s): "
            f"{[str(d) for d in args.client_devices]}. "
            f"WARNING: threaded mode is NOT deterministic — do not use for canonical paper runs."
        )
    else:
        args.client_devices = [device]
        if not hasattr(args, 'num_parallel_clients') or args.num_parallel_clients is None:
            args.num_parallel_clients = 1

    wandb_init_params = {
        "project": args.wandb_project_name,
        "name": args.wandb_run_name,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "config": args,
    }
    if args.wandb_run_id_resume:
        wandb_init_params["resume"] = "allow"
        wandb_init_params["id"] = args.wandb_run_id_resume
        logging.info(f"Attempting to resume W&B run with ID: {args.wandb_run_id_resume}")
    else:
        logging.info("Starting a new W&B run.")
    wandb.init(**wandb_init_params)
    
    # Set 'round' as the default x-axis for all metrics
    wandb.define_metric("round")
    wandb.define_metric("*", step_metric="round")

    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)

    # Pre-compute client budgets BEFORE load_data when correlated partitioning is enabled.
    # This must run after seeding so the result is reproducible, and before load_data so
    # the budgets are available during the Dirichlet partition step.
    args._pre_computed_budgets = None
    if getattr(args, 'weight_dataset_by_budget', False) and args.resource_heterogeneity:
        logging.info(
            f"[Correlated Data] Pre-computing client budgets before data partition "
            f"(gamma={args.corr_gamma})."
        )
        args._pre_computed_budgets = _precompute_client_budgets(args)
        logging.info(
            f"[Correlated Data] Budgets pre-computed for {len(args._pre_computed_budgets)} clients."
        )

    # Load data
    args.device = device
    dataset = load_data(args, args.dataset)
    
    # Set number of classes from the dataset
    # Dataset array format (10 elements): [train_num, val_num, test_num, train_global, val_global, test_global, local_num_dict, train_local_dict, test_local_dict, class_num]
    args.num_classes = dataset[9]

    # Collect all supernet architecture parameters into a dictionary
    arch_params = {
        'num_stages': args.supernet_num_stages,
        'initial_input_hw': args.supernet_initial_input_hw,
        'initial_input_channels': args.supernet_initial_input_channels,
        'stem_stride': args.supernet_stem_stride,
        'original_stem_out_channels': args.supernet_original_stem_out_channels,
        'original_stage_base_channels': args.supernet_original_stage_base_channels,
        'stage_downsample_factors': args.supernet_stage_downsample_factors,
        'max_extra_blocks_per_stage': args.supernet_max_extra_blocks_per_stage,
        'channel_divisible_by': args.supernet_channel_divisible_by,
        'width_multiplier_choices': args.supernet_width_multiplier_choices,
        'expansion_ratio_choices': args.supernet_expansion_ratio_choices,
        'n_classes': args.num_classes,
        'n_classes': args.num_classes,
        'bn_gamma_zero_init': args.bn_gamma_zero_init,
        # Resource Heterogeneity
        'resource_heterogeneity': args.resource_heterogeneity,
        'resource_distribution_type': args.resource_distribution_type,
        'resource_zipf_alpha': args.resource_zipf_alpha,
        'resource_min_mac': args.resource_min_mac,
        'resource_max_mac': args.resource_max_mac,
        'subnet_cache_path': args.subnet_cache_path,
        'resource_force_max_clients': args.resource_force_max_clients,
        # Classifier dropout (overfitting mitigation)
        'dropout_rate': getattr(args, 'classifier_dropout', 0.0),
    }
    logger.debug(f"Assembled Architecture Parameters: {arch_params}")


    # --- Check for Resume Round from Checkpoint Metadata ---
    # We must do this BEFORE create_model because create_model uses args.resume_round 
    # to decide whether to restore RNG states.
    checkpoint_metadata_round = None
    checkpoint_best_accuracy = None
    if args.local_model_ckpt_path and os.path.exists(args.local_model_ckpt_path):
        try:
             # Just peek at the keys
             # We must set weights_only=False because the checkpoint contains numpy RNG states and other metadata
             ckpt = torch.load(args.local_model_ckpt_path, map_location='cpu', weights_only=False)
             if isinstance(ckpt, dict) and "round" in ckpt:
                 checkpoint_metadata_round = ckpt["round"] + 1 # Resume from NEXT round
                 logging.info(f"Detected round {ckpt['round']} in checkpoint. Next round: {checkpoint_metadata_round}")
             if isinstance(ckpt, dict) and "best_accuracy" in ckpt:
                 checkpoint_best_accuracy = ckpt["best_accuracy"]
                 logging.info(f"Detected best_accuracy={checkpoint_best_accuracy:.4f} in checkpoint.")
        except Exception as e:
            logging.info(f"Could not read metadata from checkpoint: {e}")

    actual_start_round = 0
    if args.resume_round > 0:
        actual_start_round = args.resume_round
        logging.info(f"Resuming training from explicit arg: round {actual_start_round}")
    elif checkpoint_metadata_round is not None:
        actual_start_round = checkpoint_metadata_round
        # IMPORTANT: Update args.resume_round so create_model consumes it!
        args.resume_round = actual_start_round 
        logging.info(f"Auto-resuming training from checkpoint metadata: round {actual_start_round}")
    else:
        logging.info("Starting training from scratch (round 0).")


    # The --model argument acts as a switch for which type of generic model to use.
    server_model = create_model(args, output_dim=args.num_classes, device=device)

    if args.wandb_watch:
        logging.warning("Watching model parameters")
        wandb.watch(
            server_model.model, log="parameters", log_freq=args.wandb_watch_freq,
        )

    # Client trainer setup
    client_trainer_params = {
        "model": None, # Client model is set per-round
        "device": device,
        "args": args
    }
    
    # Server trainer setup
    server_trainer_params = {
        "server_model": server_model,
        "dataset": dataset,
        "args": args,
        "start_round": actual_start_round,
        "best_accuracy": checkpoint_best_accuracy,
    }

    # Teacher model setup (skip for in-place KD strategies)
    teacher_model = None
    if args.kd_ratio > 0 and args.training_strategy not in inplace_kd_strategies:
        # Assuming teacher model uses the same architecture
        teacher_model = create_model(
            args, arch_params, output_dim=args.num_classes, device=device, load_teacher=True
        )
        server_trainer_params["teacher_model"] = teacher_model
        client_trainer_params["teacher_model"] = teacher_model

    server_trainer_params["client_trainer"] = custom_client_trainer(client_trainer_params)
    
    # LR scheduler and weighted-average schedule setup
    flofa_lr_scheduler = None
    if getattr(args, 'lr_cosine', False):
        import math

        class CosineLRScheduler:
            """Cosine decay from args.lr to 0 over total_rounds FL rounds."""
            def __init__(self, base_lr, total_rounds):
                self.base_lr = base_lr
                self.total_rounds = max(total_rounds, 1)

            def get_lr(self, round_num):
                return self.base_lr * 0.5 * (1 + math.cos(math.pi * round_num / self.total_rounds))

        flofa_lr_scheduler = CosineLRScheduler(args.lr, args.comm_round)
        logging.info(f"Cosine LR scheduler: {args.lr} → 0 over {args.comm_round} rounds")
    elif args.lr_schedule and args.lr_schedule.get("type"):
        # Not implemented: this branch is a no-op, so flofa_lr_scheduler stays
        # None and training silently falls back to constant args.lr. Use
        # --lr_cosine for a scheduled LR instead.
        pass
    server_trainer_params["lr_scheduler"] = flofa_lr_scheduler

    server_trainer_params["wt_avg_sched_method"] = "Uniform"
    if args.weighted_avg_schedule and args.weighted_avg_schedule.get("type"):
        server_trainer_params["wt_avg_sched_method"] = args.weighted_avg_schedule["type"]

    # Instantiate and start the training
    server_trainer = custom_server_trainer(server_trainer_params)
    server_trainer.train()