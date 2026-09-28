#!/usr/bin/env python3
"""Reproduce the deterministic FEAST theory constants.

The utility reads the canonical client partition and, optionally, the saved
4000-round client schedule. It verifies the routed-envelope chain and computes
the finite-population coverage probability pi_u and auxiliary routed
computation scale psi_u defined in the theoretical appendix. It writes a
profile table, JSON summary, and LaTeX macros for these quantities.

Consumes the exact output schema of scripts/theory_constants/dump_canonical_partition.py.

Example
-------
python scripts/theory_constants/dump_canonical_partition.py \
  --out results/cifar100/canonical_partition.csv \
  --schedule-out results/cifar100/canonical_round_schedule.csv

python scripts/theory_constants/reproduce_theory_constants.py \
  --partition results/cifar100/canonical_partition.csv \
  --schedule results/cifar100/canonical_round_schedule.csv \
  --out-dir results/cifar100/theory_constants
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PARTITION_COLUMNS = {
    "client",
    "n_i",
    "B_i_batches",
    "envelope_id",
    "envelope_max_d",
    "envelope_max_w_indices",
    "n_affordable",
    "local_max_cache_idx",
}

REQUIRED_SCHEDULE_COLUMNS = {
    "round",
    "eta_t",
    "client",
    "n_i",
    "B_i",
    "envelope_id",
}


def parse_vector(value: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError(f"Expected an integer-list string, got: {value!r}")
    return tuple(parsed)


def choose_or_zero(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def validate_partition(df: pd.DataFrame, batch_size: int) -> None:
    missing = REQUIRED_PARTITION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Partition CSV is missing columns: {sorted(missing)}")
    if df["client"].duplicated().any():
        raise ValueError("Partition CSV contains duplicate client IDs")
    expected = np.ceil(df["n_i"].to_numpy(dtype=float) / batch_size).astype(int)
    actual = df["B_i_batches"].to_numpy(dtype=int)
    if not np.array_equal(expected, actual):
        bad = df.loc[expected != actual, ["client", "n_i", "B_i_batches"]]
        raise ValueError(f"B_i_batches != ceil(n_i/{batch_size}) for:\n{bad}")
    if (df["n_i"] <= 0).any():
        raise ValueError("All clients must have positive sample counts")


def ordered_envelopes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    envelopes = df.drop_duplicates("envelope_id").copy()
    envelopes["envelope_vector"] = envelopes.apply(
        lambda row: parse_vector(row["envelope_max_d"])
        + parse_vector(row["envelope_max_w_indices"]),
        axis=1,
    )
    envelopes["vector_sum"] = envelopes["envelope_vector"].map(sum)
    envelopes = envelopes.sort_values(
        ["vector_sum", "n_affordable", "local_max_cache_idx", "envelope_id"]
    ).reset_index(drop=True)

    vectors = envelopes["envelope_vector"].tolist()
    for lower, upper in zip(vectors, vectors[1:]):
        if len(lower) != len(upper) or not all(x <= y for x, y in zip(lower, upper)):
            raise ValueError("Routed envelopes are not a componentwise-nested chain")

    rank = {int(env_id): idx for idx, env_id in enumerate(envelopes["envelope_id"])}
    return envelopes, rank


def subset_count_dp(sample_counts: np.ndarray, max_size: int) -> np.ndarray:
    """Return F[s,w] = number of subsets of size s and sample-count sum w."""
    total = int(sample_counts.sum())
    dp = np.zeros((max_size + 1, total + 1), dtype=np.int64)
    dp[0, 0] = 1
    current_sum = 0
    processed = 0
    for value in sample_counts.astype(int):
        processed += 1
        current_sum += value
        for size in range(min(max_size, processed), 0, -1):
            dp[size, value : current_sum + 1] += dp[
                size - 1, : current_sum + 1 - value
            ]
    return dp


def exact_profile_statistics(
    population: pd.DataFrame,
    routed_indices: np.ndarray,
    cohort_size: int,
) -> dict[str, Any]:
    """Compute exact beta weights, coverage, and psi for one routed profile."""
    n_clients = len(population)
    routed = population.loc[routed_indices]
    sample_counts = routed["n_i"].to_numpy(dtype=int)
    local_steps = routed["B_i_batches"].to_numpy(dtype=float)
    routed_count = len(routed)
    total_mass = int(sample_counts.sum())
    denominator = math.comb(n_clients, cohort_size)

    max_subset_size = cohort_size - 1
    full_dp = subset_count_dp(sample_counts, max_subset_size)
    betas: list[float] = []

    for n_i in sample_counts:
        # Polynomial division:
        # F_s(w) = D_s^{(-i)}(w) + D_{s-1}^{(-i)}(w-n_i).
        previous = np.zeros(total_mass + 1, dtype=np.int64)
        previous[0] = 1
        beta_numerator = 0.0

        coefficient = choose_or_zero(n_clients - routed_count, cohort_size - 1)
        if coefficient:
            beta_numerator += coefficient

        for subset_size in range(1, max_subset_size + 1):
            leave_one_out = full_dp[subset_size].copy()
            leave_one_out[n_i:] -= previous[:-n_i]
            coefficient = choose_or_zero(
                n_clients - routed_count,
                cohort_size - subset_size - 1,
            )
            if coefficient:
                nonzero_sums = np.flatnonzero(leave_one_out)
                beta_numerator += coefficient * n_i * np.sum(
                    leave_one_out[nonzero_sums].astype(np.float64)
                    / (n_i + nonzero_sums)
                )
            previous = leave_one_out

        betas.append(beta_numerator / denominator)

    beta = np.asarray(betas, dtype=float)
    if n_clients - routed_count >= cohort_size:
        coverage = 1.0 - math.comb(n_clients - routed_count, cohort_size) / denominator
    else:
        coverage = 1.0

    psi = float(np.dot(beta, local_steps))
    varpi = beta * local_steps / psi

    return {
        "routed_clients": routed_count,
        "coverage_probability": coverage,
        "computation_scale": psi,
        "beta_sum": float(beta.sum()),
        "varpi_sum": float(varpi.sum()),
    }


def validate_schedule(
    schedule: pd.DataFrame,
    partition: pd.DataFrame,
    rounds: int,
    cohort_size: int,
    eta0: float,
) -> None:
    missing = REQUIRED_SCHEDULE_COLUMNS - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule CSV is missing columns: {sorted(missing)}")
    counts = schedule.groupby("round").size()
    if len(counts) != rounds or not (counts == cohort_size).all():
        raise ValueError("Schedule must contain exactly cohort_size rows for every round")
    expected_eta = eta0 * 0.5 * (
        1.0 + np.cos(np.pi * schedule["round"].to_numpy(dtype=float) / rounds)
    )
    if not np.allclose(schedule["eta_t"], expected_eta, rtol=0.0, atol=5e-13):
        raise ValueError("Schedule eta_t does not match the canonical cosine formula")

    lookup = partition.set_index("client")
    joined = schedule.join(
        lookup[["n_i", "B_i_batches", "envelope_id"]],
        on="client",
        rsuffix="_partition",
    )
    checks = {
        "n_i": "n_i_partition",
        "B_i": "B_i_batches",
        "envelope_id": "envelope_id_partition",
    }
    for left, right in checks.items():
        if not np.array_equal(joined[left].to_numpy(), joined[right].to_numpy()):
            raise ValueError(f"Schedule column {left} disagrees with the partition")


def realized_profile_statistics(
    schedule: pd.DataFrame,
    population: pd.DataFrame,
    threshold_rank: int,
) -> tuple[float, float]:
    routed_clients = set(
        population.loc[population["envelope_rank"] >= threshold_rank, "client"].astype(int)
    )
    routed = schedule[schedule["client"].isin(routed_clients)].copy()
    denominator = routed.groupby("round")["n_i"].sum()
    numerator = (routed["n_i"] * routed["B_i"]).groupby(routed["round"]).sum()
    all_rounds = np.arange(schedule["round"].min(), schedule["round"].max() + 1)
    denominator = denominator.reindex(all_rounds, fill_value=0.0)
    numerator = numerator.reindex(all_rounds, fill_value=0.0)
    covered = denominator.to_numpy() > 0
    realized_psi = np.zeros_like(denominator.to_numpy(dtype=float))
    realized_psi[covered] = numerator.to_numpy(dtype=float)[covered] / denominator.to_numpy(dtype=float)[covered]
    return float(covered.mean()), float(realized_psi.mean())


def latex_macros(summary: dict[str, Any]) -> str:
    macros = {
        "TheoryClientCount": summary["client_count"],
        "TheorySampleTotal": summary["sample_total"],
        "TheorySampleMin": summary["sample_min"],
        "TheorySampleMax": summary["sample_max"],
        "TheoryLocalStepMin": summary["local_step_min"],
        "TheoryLocalStepMax": summary["local_step_max"],
        "TheoryEnvelopeCount": summary["envelope_count"],
        "TheoryRMin": summary["r_min"],
        "TheoryRMax": summary["r_max"],
        "TheoryPiMin": f"{summary['pi_min']:.9f}",
        "TheoryPiMax": f"{summary['pi_max']:.9f}",
        "TheoryPsiMin": f"{summary['psi_min']:.9f}",
        "TheoryPsiMax": f"{summary['psi_max']:.9f}",
    }
    if summary.get("max_realized_coverage_deviation") is not None:
        macros["TheoryCoverageAuditDeviation"] = (
            f"{summary['max_realized_coverage_deviation']:.5f}"
        )
        macros["TheoryPsiAuditDeviation"] = f"{summary['max_realized_psi_deviation']:.5f}"
    return "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cohort-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--eta0", type=float, default=0.025)
    args = parser.parse_args()

    partition = pd.read_csv(args.partition).sort_values("client").reset_index(drop=True)
    validate_partition(partition, args.batch_size)
    envelopes, rank = ordered_envelopes(partition)
    partition["envelope_rank"] = partition["envelope_id"].map(rank).astype(int)

    profile_rows: list[dict[str, Any]] = []
    for threshold_rank, envelope in envelopes.iterrows():
        routed_indices = partition.index[partition["envelope_rank"] >= threshold_rank].to_numpy()
        stats = exact_profile_statistics(partition, routed_indices, args.cohort_size)
        row: dict[str, Any] = {
            "threshold_rank": int(threshold_rank),
            "threshold_envelope_id": int(envelope["envelope_id"]),
            "threshold_depth": envelope["envelope_max_d"],
            "threshold_width_indices": envelope["envelope_max_w_indices"],
            **stats,
        }
        profile_rows.append(row)

    profile_table = pd.DataFrame(profile_rows)

    schedule = None
    if args.schedule is not None:
        schedule = pd.read_csv(args.schedule)
        validate_schedule(schedule, partition, args.rounds, args.cohort_size, args.eta0)
        for idx, row in profile_table.iterrows():
            realized_pi, realized_psi = realized_profile_statistics(
                schedule, partition, int(row["threshold_rank"])
            )
            profile_table.loc[idx, "realized_coverage"] = realized_pi
            profile_table.loc[idx, "realized_computation_scale"] = realized_psi
            profile_table.loc[idx, "coverage_deviation"] = (
                realized_pi - row["coverage_probability"]
            )
            profile_table.loc[idx, "computation_scale_deviation"] = (
                realized_psi - row["computation_scale"]
            )

    psi_max = float(profile_table["computation_scale"].max())
    summary: dict[str, Any] = {
        "client_count": int(len(partition)),
        "sample_total": int(partition["n_i"].sum()),
        "sample_min": int(partition["n_i"].min()),
        "sample_max": int(partition["n_i"].max()),
        "local_step_total": int(partition["B_i_batches"].sum()),
        "local_step_min": int(partition["B_i_batches"].min()),
        "local_step_max": int(partition["B_i_batches"].max()),
        "envelope_count": int(len(envelopes)),
        "r_min": int(profile_table["routed_clients"].min()),
        "r_max": int(profile_table["routed_clients"].max()),
        "pi_min": float(profile_table["coverage_probability"].min()),
        "pi_max": float(profile_table["coverage_probability"].max()),
        "psi_min": float(profile_table["computation_scale"].min()),
        "psi_max": psi_max,
    }
    if schedule is not None:
        summary.update(
            {
                "schedule_rows": int(len(schedule)),
                "realized_total_local_steps": int(schedule["B_i"].sum()),
                "max_realized_coverage_deviation": float(
                    profile_table["coverage_deviation"].abs().max()
                ),
                "max_realized_psi_deviation": float(
                    profile_table["computation_scale_deviation"].abs().max()
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile_table.to_csv(args.out_dir / "theory_profile_constants.csv", index=False)
    (args.out_dir / "theory_constants.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "theory_constants_macros.tex").write_text(latex_macros(summary))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
