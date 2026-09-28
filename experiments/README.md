# experiments/

Every experiment in the paper is a self-contained shell script that calls
`train.py` (for FEAST/SuperFedNAS/DeepFedNAS) or a baseline's standalone
trainer module (`python -m baselines.<method>.trainer`, for
HeteroFL/ScaleFL/FIARSE) with every hyperparameter spelled out explicitly on
the command line — there is no shared config-file indirection, so any two
scripts can be diffed directly to see exactly what varies between them. Most
scripts carry long inline comments explaining *why* each flag has the value
it does (compute-cost decisions, protocol history, faithfulness notes) —
read the script itself for that, this README only covers what each script
*is*.

## Subfolders (in the order they appear in the paper's narrative)

- **`01_superfednas/`** — SuperFedNAS: the naive single-subnet-per-client +
  real MaxNet cosine-annealed aggregation baseline this paper's training
  approach (FEAST) replaces. Uniform-random subnet sampling.
- **`02_deepfednas/`** — DeepFedNAS: same MaxNet aggregation mechanism as
  SuperFedNAS, but with Pareto-path-guided subnet sampling instead of
  uniform-random. Both are kept as negative baselines demonstrating that
  MaxNet aggregation collapses under hard non-IID FL — neither is FEAST.
- **`03_cifar100/`**, **`04_cinic10/`**, **`05_tinyimagenet/`** — FEAST's
  canonical main-result run plus the three external baselines
  (HeteroFL/ScaleFL/FIARSE) at matched protocol, one folder per dataset.
- **`06_feast_ablations/`** — ablates FEAST's own per-step min/random/max
  training rule (min/random/local-max steps + KD) to isolate which
  component matters.
- **`07_gamma_ablation/`** — sweeps the compute-data correlation parameter
  gamma across FEAST and all three baselines (gamma=1 canonical lives in
  03/04/05; this folder covers gamma=0/0.25/0.5/1.5).
- **`08_zipf_sensitivity/`** — sweeps the client-compute Zipf(alpha)
  parameter for FEAST only.
- **`09_dirichlet_sensitivity/`** — sweeps the label-distribution Dirichlet
  alpha for FEAST and FIARSE only (not HeteroFL/ScaleFL).
- **`10_matched_training_compute/`** — extends each baseline's round count
  so its *total* training MACs match FEAST's canonical run, controlling for
  the objection "FEAST just trained longer." Baselines only — FEAST's own
  matched-compute run is just its canonical run in `03_cifar100/`.
