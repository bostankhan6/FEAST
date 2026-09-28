# FEAST: Federated Shared-Space Training for Resource-Heterogeneous Clients

[Read the paper on arXiv](https://arxiv.org/abs/2608.09250).

FEAST trains a shared OFA-style supernet across resource-heterogeneous
federated clients (CIFAR-100, CINIC-10, TinyImageNet) with a training recipe
that produces a flat accuracy-vs-compute curve across the full client compute
range. Three components:

1. **Local Multi-variant Co-training** — global-min / random /
   local-max subnet per step with knowledge distillation, preventing
   gradient interference on shared supernet parameters.
2. **γ-correlated compute-data allocation** — positive coupling between
   client compute budget and local dataset size (γ=1 default), fixing the
   data-access asymmetry that degrades large-subnet accuracy under
   uncorrelated (γ=0) allocation.
3. **Sub-supernet communication** — server sends only the parameter slice
   needed for a client's affordable subnets, reducing bandwidth relative to
   full-supernet broadcast.

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{khan2026feast,
  title={FEAST: Federated Shared-Space Training for Resource-Heterogeneous Clients},
  author={Bostan Khan and Masoud Daneshtalab},
  year={2026},
  eprint={2608.09250},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2608.09250}
}
```

## Repository structure

```
src/feast/               core federated NAS system
  Server/                server model, aggregation, sub-supernet logic
  Client/                client trainer
  elastic_nn/             OFA supernet (GenericOFAResNet)
  data/                   CIFAR-100, CINIC-10, TinyImageNet loaders and partitioning
  nas/                    genetic-algorithm fitness search (cached + post-hoc subnet discovery)
  utils/                  analytic MACs/params costing (subnet_cost.py)

baselines/                faithful external baseline implementations
  heterofl/                HeteroFL (ICLR 2021)
  scalefl/                 ScaleFL (CVPR 2023)
  fiarse/                  FIARSE (NeurIPS 2024)

configs/supernets/        supernet architecture-search-space configs
experiments/              experiment entry-point shell scripts, one folder per experiment group
scripts/                  data setup, subnet-cache generation, evaluation, centralized baselines
subnet_caches/            pre-computed subnet MAC/config tables
tests/                    unit and integration tests
train.py                  main federated training entry point
parse_args.py             CLI argument definitions
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Requires Python 3.8+, PyTorch 1.13+.

## Data

```bash
bash scripts/data_setup/download_cifar100.sh
bash scripts/data_setup/download_cinic10.sh
python scripts/data_setup/prepare_tinyimagenet.py   # after downloading tiny-imagenet-200.zip
```

CIFAR-100's loader can also auto-download via torchvision on first run, but
CINIC-10 (Edinburgh DataShare) has no programmatic download support, so both
are fetched by the setup scripts above for consistency. See `data/README.md`
for details and `src/feast/data/` for the loaders.

## Training

Each experiment under `experiments/` is a self-contained shell script that calls
`train.py` with the relevant flags. Example (FEAST, CIFAR-100):

```bash
bash experiments/03_cifar100/FEAST_cifar100.sh
```

`experiments/01_superfednas/` and `experiments/02_deepfednas/` contain the
naive single-subnet-per-client + plain-FedAvg negative baseline (labeled
SuperFedNAS / DeepFedNAS in the paper — the prior training approach FEAST
replaces, not FEAST itself). `baselines/` contains the external methods
FEAST is compared against.

## Checkpoints

Every `train.py`-based experiment script (FEAST itself, plus the
`01_superfednas`/`02_deepfednas` baselines) passes `--checkpoint_dir
checkpoints/<run-name>`, which is saved to independently of wandb — training
still works with `WANDB_MODE=disabled` or no wandb account at all. The best
checkpoint is `best_checkpoint_supernet.pt` inside that directory. (If
`--checkpoint_dir` is omitted, checkpoints are only saved via wandb, into
`wandb.run.dir`, and are lost entirely if wandb is disabled.)

`baselines/{heterofl,scalefl,fiarse}/trainer.py` always save locally to
`--checkpoint_dir` (default `checkpoints/<method>`), independent of wandb —
`best_full.pt` and `checkpoint_latest.pt`.

## Evaluation

`scripts/evaluate.py` builds a `GenericServerOFA` model from the checkpoint's
`arch_params`, so it only loads checkpoints produced by FEAST itself or the
SuperFedNAS/DeepFedNAS baselines (all three share the same OFA supernet
class) — not HeteroFL/ScaleFL/FIARSE, which use separate model builders under
`baselines/`. Pass `--subnet_cache` to evaluate every subnet in that cache (or
a filtered subset via `--subnet_indices`); without `--subnet_cache` it
defaults to evaluating just the min and max subnets.

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/feast-cifar100-mixaug/best_checkpoint_supernet.pt \
  --dataset cifar100 \
  --subnet_cache subnet_caches/<cache>.csv \
  --subnet_indices 0 3 22 38 55 \
  --augmentation mixaug \
  --output <path/to/eval_output.csv>
```

`scripts/evaluate_compare.py` evaluates and compares FEAST against
HeteroFL/ScaleFL/FIARSE at the same subnet indices, given each method's own
checkpoint.

## Population-weighted accuracy and cross-method comparisons

`scripts/evaluate.py`/`evaluate_compare.py` report raw per-subnet accuracy.
The paper's headline numbers are population-weighted: each of N clients is
drawn from the canonical Zipf(alpha) compute-budget distribution and scored
by the best subnet/tier it can afford. `scripts/evaluate_pop_weighted.py`
computes this from per-method evaluation CSVs (i.e. the `--output` of
`evaluate.py`, run over the full subnet cache):

```bash
python scripts/evaluate_pop_weighted.py \
  --method FEAST:results/cifar100/feast_eval.csv \
  --method HeteroFL:results/cifar100/heterofl_eval.csv \
  --method ScaleFL:results/cifar100/scalefl_eval.csv \
  --method FIARSE:results/cifar100/fiarse_eval.csv \
  --baseline FEAST \
  --out results/cifar100/pop_weighted_summary.csv
```

`scripts/client_param_footprint.py` computes the client-side parameter count
each method actually transmits at matched compute budgets — the basis for
any bandwidth/communication-cost comparison:

```bash
python scripts/client_param_footprint.py \
  --out results/cifar100/client_param_footprint.csv
```

## Reproducing the sub-supernet communication reduction

`scripts/reproduce_sub_supernet_communication_reduction.py` validates the
routed sub-supernet mechanism under the canonical Zipf(1.2) client-budget
distribution (every cached subnet activates correctly, routing bounds are
exactly tight, forward passes succeed) and reports the population-weighted
communication reduction vs. broadcasting the full supernet — the number the
paper cites for that claim:

```bash
python scripts/reproduce_sub_supernet_communication_reduction.py
```

`--synthetic` runs the same checks against a fast synthetic cache instead of
the real one (for a quick sanity check; doesn't reproduce the paper's
number). `tests/test_sub_supernet_zipf_validation.py` runs the `--synthetic`
path under pytest as part of CI.

## Realized canonical experimental artifacts

The protocol descriptions elsewhere (client budget generation, cache
construction) are stochastic procedures. The concrete realizations they
produce for the canonical CIFAR-100 run are all present in this repo:

| Item | Where |
|---|---|
| Realized client budget vector | `results/cifar100/canonical_partition.csv` — `budget_M`/`capped_budget_M` per client (regenerate via the command below) |
| Realized client sample counts / partition manifest | Same file — `n_i`, `B_i_batches`, plus routing info (`envelope_id`, `envelope_max_d`, `envelope_max_w_indices`, `n_affordable`) per client |
| All 56 cached architecture configs and their MACs | `subnet_caches/extended_range_25M_1500M.csv` — `d`, `e`, `w_indices`, `macs`, `params` per row |
| Exact cache targets and random seeds | Same file — `job_target_macs`, `job_seed` columns per row |
| Participation schedule | `results/cifar100/canonical_round_schedule.csv` — one row per (round, client) for all 4000 rounds x 10 clients/round (regenerate via the command below); the generating rule itself is `np.random.seed(round_idx)` then `np.random.choice(N, 10, replace=False)`, so it is also re-derivable from the round index alone |

`results/` is gitignored (generated output, like `checkpoints/`) — regenerate
the partition and schedule files with:

```bash
python scripts/theory_constants/dump_canonical_partition.py \
  --out results/cifar100/canonical_partition.csv \
  --schedule-out results/cifar100/canonical_round_schedule.csv
```

This is deterministic: seed=0, 100 clients, Zipf(1.2), budget range
[25M, 1.5G], cap 600M, partition alpha=0.1 — the canonical protocol's own
defaults.

## Reproducing the theory constants

`scripts/theory_constants/reproduce_theory_constants.py` computes the
finite-population coverage probability and routed computation scale used in
the paper's theoretical appendix, from the canonical client partition and
round schedule. See `scripts/theory_constants/README.md` for the full
command and what each output means.

## Subnet cache generation

The canonical cache (`subnet_caches/extended_range_25M_1500M.csv`, 56 subnets,
25M-596M MACs) is a merge of two genetic-algorithm searches over
`configs/supernets/4-stage-supernet-cifar100-v2.json`:

```bash
bash scripts/cache_generation/build_canonical_cache.sh
```

This runs `generate_cache_main_segment.sh` (50M-600M, 50 subnets), then
`generate_cache_low_end_segment.sh` (25M-75M, 6 subnets), then
`merge_extended_caches.py` to combine them. Each step can also be run
individually.

The same fitness search (`src/feast/nas/feast_fitness_maximizer.py`) is also
used for post-hoc/uncached architecture discovery at deployment time — see
`scripts/post_hoc_extraction/`.

## License

MIT (see `pyproject.toml`).
