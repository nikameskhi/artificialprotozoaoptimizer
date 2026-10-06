"""Run reproducible, matched-seed comparisons on the spring-design problem."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from apo import APO_orig
from apo_de import APO_DE
from apo_ls import APO_LS
from apo_pso import APO_PSO
from apo_wr import APO_wr
from apo_wr_de import APO_wr_DE
from apo_wr_ls import APO_wr_LS
from apo_wr_pso import APO_wr_PSO
from spring_design import LOWER_BOUNDS, UPPER_BOUNDS, SpringDesign


ALGORITHMS = (
    ("APO", APO_orig),
    ("APO-wr", APO_wr),
    ("APO-DE", APO_DE),
    ("APO-LS", APO_LS),
    ("APO-PSO", APO_PSO),
    ("APO-wr-DE", APO_wr_DE),
    ("APO-wr-LS", APO_wr_LS),
    ("APO-wr-PSO", APO_wr_PSO),
)
NEIGHBOR_PAIRS = 1
PF_MAX = 0.1


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_rows: list[dict]) -> list[dict]:
    rows = []
    for name, _ in ALGORITHMS:
        records = [row for row in run_rows if row["algorithm"] == name]
        fitness = np.array([row["penalized_fitness"] for row in records])
        rows.append(
            {
                "algorithm": name,
                "best_penalized": float(np.min(fitness)),
                "median_penalized": float(np.median(fitness)),
                "mean_penalized": float(np.mean(fitness)),
                "std_penalized": float(np.std(fitness, ddof=1)) if len(fitness) > 1 else 0.0,
                "worst_penalized": float(np.max(fitness)),
                "mean_raw_objective": float(
                    np.mean([row["raw_objective"] for row in records])
                ),
                "mean_constraint_violation": float(
                    np.mean([row["total_constraint_violation"] for row in records])
                ),
                "feasible_rate": float(np.mean([row["feasible"] for row in records])),
            }
        )
    return rows


def paired_statistics(run_rows: list[dict]) -> list[dict]:
    by_algorithm_seed = {
        (row["algorithm"], row["seed"]): row["penalized_fitness"]
        for row in run_rows
    }
    comparisons = []
    for first, second in combinations((name for name, _ in ALGORITHMS), 2):
        first_values = np.array(
            [
                by_algorithm_seed[(first, seed)]
                for seed in sorted(
                    seed for algorithm, seed in by_algorithm_seed
                    if algorithm == first
                )
            ]
        )
        seeds = sorted(
            seed for algorithm, seed in by_algorithm_seed if algorithm == first
        )
        second_values = np.array(
            [by_algorithm_seed[(second, seed)] for seed in seeds]
        )
        differences = first_values - second_values
        if np.allclose(differences, 0, rtol=0, atol=1e-15):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(first_values, second_values, alternative="two-sided")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        comparisons.append(
            {
                "comparison": f"{first} vs {second}",
                "wilcoxon_statistic": statistic,
                "raw_p_value": p_value,
            }
        )

    # Holm step-down adjustment controls family-wise error across these pairs.
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["raw_p_value"])
    adjusted = [0.0] * len(comparisons)
    running_max = 0.0
    total = len(comparisons)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * comparisons[index]["raw_p_value"])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    for row, p_adjusted in zip(comparisons, adjusted):
        row["holm_p_value"] = p_adjusted
        row["significant_0_05"] = int(p_adjusted < 0.05)
    return comparisons


def run(trials: int, population: int, iterations: int, seed_start: int, seed_step: int,
        output_dir: Path) -> None:
    if trials < 1 or population < 4 or iterations < 2 or seed_step < 1:
        raise ValueError("trials >= 1, population >= 4, iterations >= 2, and seed-step >= 1 are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    objective = SpringDesign()
    run_rows = []
    trace_rows = []

    for trial in range(1, trials + 1):
        seed = seed_start + (trial - 1) * seed_step
        for name, algorithm in ALGORITHMS:
            np.random.seed(seed)
            trace, position, _ = algorithm(
                objective,
                len(LOWER_BOUNDS),
                population,
                iterations,
                LOWER_BOUNDS,
                UPPER_BOUNDS,
                NEIGHBOR_PAIRS,
                PF_MAX,
            )
            penalized, raw_objective, violation = objective.components(position)
            run_rows.append(
                {
                    "algorithm": name,
                    "trial": trial,
                    "seed": seed,
                    "penalized_fitness": penalized,
                    "raw_objective": raw_objective,
                    "total_constraint_violation": violation,
                    "feasible": int(violation <= 1e-8),
                    "x1": float(position[0]),
                    "x2": float(position[1]),
                    "x3": float(position[2]),
                }
            )
            for generation, fitness in enumerate(trace):
                trace_rows.append(
                    {
                        "algorithm": name,
                        "trial": trial,
                        "seed": seed,
                        "evaluations": population * (generation + 1),
                        "best_penalized_fitness": float(fitness),
                    }
                )
            print(
                f"trial {trial}/{trials} {name}: "
                f"fitness={penalized:.10g}, feasible={violation <= 1e-8}"
            )

    write_csv(
        output_dir / "runs.csv",
        [
            "algorithm", "trial", "seed", "penalized_fitness", "raw_objective",
            "total_constraint_violation", "feasible", "x1", "x2", "x3",
        ],
        run_rows,
    )
    write_csv(
        output_dir / "convergence.csv",
        [
            "algorithm", "trial", "seed", "evaluations",
            "best_penalized_fitness",
        ],
        trace_rows,
    )
    write_csv(
        output_dir / "summary.csv",
        [
            "algorithm", "best_penalized", "median_penalized", "mean_penalized",
            "std_penalized", "worst_penalized", "mean_raw_objective",
            "mean_constraint_violation", "feasible_rate",
        ],
        summarize(run_rows),
    )
    write_csv(
        output_dir / "statistics.csv",
        [
            "comparison", "wilcoxon_statistic", "raw_p_value",
            "holm_p_value", "significant_0_05",
        ],
        paired_statistics(run_rows),
    )
    print(f"Wrote experiment CSVs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2022)
    parser.add_argument("--seed-step", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results_all"))
    args = parser.parse_args()
    run(
        args.trials,
        args.population,
        args.iterations,
        args.seed_start,
        args.seed_step,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
