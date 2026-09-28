# `feast` package

Installable Python package (`pip install -e .`, imported as `import feast`) containing the
elastic supernet, federated client/server training logic, NAS search utilities, and dataset
loaders used by `train.py` and the pipelines under `scripts/`.

## Layout

```
feast/
├── elastic_nn/     # Elastic (OFA-style) supernet building blocks and the full network
├── Server/         # Server-side model wrapper, subnet sampling/aggregation, per-round trainer
├── Client/         # Client-side model wrapper and local training loop
├── nas/            # Architecture search: entropy-maximizing GA and subnet-cache generation
├── utils/          # Analytic MACs/params costing for a given architecture
└── data/           # Per-dataset partitioning and torch Dataset classes (CIFAR-100, CINIC-10, TinyImageNet)
```

## `elastic_nn/`

`generic_ofa_network.py` defines the elastic ResNet supernet:

- `NewResidualBlock` / `DynamicResidualBlock` — static and elastic (depth/width/expansion-ratio
  switchable) residual blocks. `DynamicResidualBlock.get_active_subnet()` extracts a static
  `NewResidualBlock` at the currently active configuration.
- `GenericStaticResNetSubnet` — a plain (non-elastic) ResNet assembled from extracted blocks;
  this is what `get_active_subnet()` on the supernet returns for training/evaluating one subnet.
- `GenericOFAResNet` — the full elastic supernet. Key methods: `set_active_subnet(d, e_indices,
  w_indices)` to switch the active architecture, `set_max_net()` / `set_min_net()` for the
  largest/smallest configuration, `sample_active_subnet()` for a uniformly random configuration,
  `get_active_subnet(preserve_weight=True)` to materialize the active subnet as a standalone
  static model, and `set_activation_checkpointing()` to trade compute for memory during the
  multi-subnet forward passes used by sandwich-style training strategies.

## `Server/`

- `base_server_model.py` — `BaseServerModel`, an abstract base class holding a `subnet_sampling`
  dict that maps a sampler name to a bound method (`static_sample`, `dynamic_sample`,
  `random_subnet_sample`, `compound_subnet_sample`, `sandwich_all_subnet_sample`,
  `sandwich_compound_subnet_sample`, `tracking_sandwich_all_subnet_sample` (`TS_all_random`),
  `tracking_sandwich_all_subnet_constrained_sample` (`TS_all_random_constrained`),
  `tracking_sandwich_entropy_maximizer` (`TS_entropy_maximizer`),
  `tracking_sandwich_optimal_path_sampler` (`TS_optimal_path`),
  `tracking_sandwich_compound_subnet_sample` (`TS_compound`), `max_client_dataset_all_subnet`
  (`max_sample_count`), `multi_sandwich_sample`, `tracking_sandwich_kd` (`TS_KD`), `ps` (`PS`)).
  `--subnet_dist_type` (`parse_args.py`) restricts the CLI to a subset of these keys
  (`static`, `TS_optimal_path`, `TS_cached_entropy_maximizer`, `TS_entropy_maximizer`, `dynamic`,
  `all_random`, `TS_all_random`, `TS_all_random_constrained`, `sandwich_all_random`, `TS_compound`,
  `max_sample_count`, `multi_sandwich`, `TS_KD`, `PS`); `compound` and `sandwich_compound` are
  registered in the dict but are not exposed as `--subnet_dist_type` choices. A further method,
  `dynamic_width_sandwich_sample`, is an unfinished `pass` stub and is not registered in the
  `subnet_sampling` dict at all. The class also owns the top-k/bottom-k client-role bookkeeping
  (`cli_subnet_track`, `update_sample`, `set_top_bottom_k`) and abstract hooks
  (`init_model`, `get_subnet`, `add_subnet`, `max_subnet_arch`, `min_subnet_arch`, ...) that
  `generic_server_model.py` implements, plus checkpoint I/O (`save`, decoupled from wandb via
  `--checkpoint_dir`).

- `generic_server_model.py` — `GenericServerOFA(BaseServerModel)`, the concrete server model
  wrapping a `GenericOFAResNet`. Responsibilities:
  - Architecture helpers: `max_subnet_arch`, `min_subnet_arch`, `random_subnet_arch`,
    `constrained_random_subnet_arch`, `random_cached_subnet_arch`,
    `random_constrained_cached_subnet_arch`, `random_strictly_constrained_cached_subnet_arch`,
    `get_client_local_max_arch` / `get_client_local_max_info` (largest cached subnet within a
    client's MAC budget), `arch_to_subnet_kwargs` (converts a `{d, e, w_indices}` dict to the
    `e_indices`-based kwargs `set_active_subnet` expects).
  - Client resource heterogeneity: `assign_client_resources` assigns each client a fixed MAC
    budget (`resource_heterogeneity`, `resource_distribution_type` = `uniform`/`zipf`,
    `resource_zipf_alpha`, `resource_min_mac`/`resource_max_mac`, `resource_force_max_clients`),
    persisted as `client_mac_budgets` and logged to `client_budgets.csv`.
    `compute_subnet_coverage` computes what fraction of clients can afford a given subnet tier,
    used for adaptive KD-ratio weighting in `Client/subnet_trainer.py`.
  - Full-supernet aggregation: `add_subnet` accumulates a client's trained weights into the
    global supernet's weighted-sum/weighted-count accumulators, handling both a full
    `GenericOFAResNet` upload and a sparse, static-subnet upload (channel-slice + key-renaming
    logic for `conv`/`bn`/`linear` layers).
  - Sub-supernet construction and aggregation (used when `--use_sub_supernet` is set, so a
    resource-constrained client only downloads/uploads a slice of the full model):
    `compute_sub_supernet_bounds` derives the max depth/width reachable within a client's budget
    from the subnet cache, `create_sub_supernet` builds a smaller `GenericOFAResNet` at those
    bounds and copies the corresponding weight slices from the global model
    (`_copy_weights_to_sub_supernet`), and `add_sub_supernet` aggregates a client's trained
    sub-supernet back into the global accumulators via the saved slice mapping.

- `feast_trainer.py` — `FeastTrainer`, the per-round orchestrator. `train()` runs the outer
  communication-round loop (client sampling, checkpointing, periodic evaluation via
  `_local_test_on_all_clients` / `_efficient_local_test_on_all_clients` /
  `_eval_global_max_on_supporters`); `train_one_round()` builds each sampled client's
  architecture bundle for the current round and dispatches training, branching on
  `--training_strategy`:
  - `standard` (default) — one architecture per client, no bundle.
  - `min_only` / `local_max_only` — every client trains a single fixed subnet (global min, or
    its own local max).
  - `feast` — per-client `{min, max, random}` bundle (max = local max within budget, random
    drawn strictly between min and max), consumed by `Client/subnet_trainer.py`'s per-step
    Max→Min / Max→Random KD training.
  - `inverse_kd_sandwich` — the same bundle shape, consumed by the alternative Min→Max /
    Max→Random KD ordering in `Client/subnet_trainer.py`.
  `_get_model_for_client` chooses between the full supernet and a constructed sub-supernet
  depending on `--use_sub_supernet`. `_aggregate` merges returned client models back into the
  server model via `add_subnet`/`add_sub_supernet`. A `weighted_avg_scheduler` dict
  (`uniform_avg`, `maxnet_linear_all_subnet`, `minnet_linear_all_subnet`,
  `maxnet_cos_all_subnet`, `maxnet_cos_all_subnet_sandwich`, `minnet_cos_all_subnet`,
  `phased_maxnet_all_subnet`, `multikd_phased_maxnet_all_subnet`) maps the `"type"` field of the
  `--weighted_avg_schedule` JSON argument to a schedule method for time-varying aggregation
  weights.

## `Client/`

- `client_model.py` — `ClientModel`, a thin wrapper around an extracted static subnet
  (`get_model`, `state_dict`, `to`/`cpu`, `set_active_subnet`, `set_max_net`, `set_avg_wt`) used
  as the unit passed between server and client during a round.
- `client_trainer.py` — `ClientTrainer(ABC)`, the base client-side trainer: local dataset
  bookkeeping (`update_local_dataset`), BN-recalibration before evaluation
  (`local_test` + `random_sub_train_loader` when `--reset_bn_stats*` is set), and the abstract
  `train`/`test` methods subclasses must implement.
- `subnet_trainer.py` — `SubnetTrainer(ClientTrainer)`, the concrete local-training loop.
  `set_model(model, arch_bundle=None, ...)` receives the architecture bundle built by
  `feast_trainer.py`; `is_sandwich_mode` is true only when `arch_bundle` is given and
  `--training_strategy` is `feast` or `inverse_kd_sandwich`. `train()` branches on
  `is_sandwich_mode` and `--training_strategy`:
  - Standard (no bundle): a single forward/backward pass per batch, with optional Mixup/CutMix
    (`_mix_batch`) and optional single-teacher KD (`--kd_ratio`, `--kd_type`).
  - `inverse_kd_sandwich`: trains Min (CE) → Max (CE + KD from a frozen Min anchor) → Random
    (CE + KD from Max), with NaN-guard logging/dumping around the Min step.
  - `feast`: trains Max (CE, becomes the in-batch teacher) → Min (CE + KD from Max) → Random
    (CE + KD from Max), optionally with adaptive per-client KD-ratio weighting
    (`--adaptive_kd`, using `coverage_stats` from `GenericServerOFA.compute_subnet_coverage`)
    and per-step random-architecture resampling (`--per_step_random`).
  - `--joint_ensemble_ce` (either sandwich strategy): trains all bundle members with one shared
    backward pass on their averaged logits instead of the sequential KD chain above.
  `--use_bn=False` freezes BatchNorm as an identity transform (SuperFedNAS-style no-BN training).

## `nas/`

- `feast_fitness_maximizer.py` — the entropy-maximizing genetic-algorithm subnet search used by
  the samplers above and by the cache-generation scripts: per-stage entropy/effectiveness
  objective (`calculate_entropy_objective`, `calculate_L_and_avg_log_w`,
  `calculate_effectiveness_rho`), a chromosome encode/decode pair (`decode_chromosome`), an
  optional latency predictor MLP (`LatencyPredictorMLP`, `predict_latency`) used when optimizing
  under a latency rather than a pure-MACs constraint, and the GA driver `run_entropy_max_ga`.
- `generate_subnet_cache.py` — batch-runs `run_entropy_max_ga` across a grid of MAC-budget
  targets to produce a subnet-cache CSV (the `subnet_cache_path` consumed by
  `TS_optimal_path`/sub-supernet code); see `scripts/cache_generation/`.
- `search_single_subnet.py` — CLI wrapper for a single GA search at one target MACs value; see
  `scripts/post_hoc_extraction/`.

## `utils/`

- `subnet_cost.py` — analytic (no forward pass) MACs and parameter counting for a given
  `(depth_vec, exp_vec, w_indices)` architecture against a supernet's `arch_config_params`;
  `make_divisible` rounds channel counts to the configured divisor. This is the shared cost
  function used by the GA fitness search, the samplers, and the evaluation scripts.

## `data/`

Three near-identical per-dataset subpackages (`cifar100/`, `cinic10/`, `tinyimagenet/`), each
with:
- `data_loader.py` — `partition_data` (hetero/Dirichlet or IID client partitioning),
  `get_dataloader_<DATASET>` / `get_dataloader_test_<DATASET>`, and
  `load_partition_data_<dataset>` (the top-level entry point `train.py` calls to obtain
  per-client train/test loaders).
- `datasets.py` — the torch `Dataset` class for that data source (e.g. `CIFAR100_truncated`),
  used internally by `data_loader.py`.
