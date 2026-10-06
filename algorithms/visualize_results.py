"""Create publication-friendly plots from the APO experiment CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "APO": "#3366a3",
    "APO-wr": "#e28e2c",
    "APO-DE": "#3a9273",
    "APO-LS": "#b64e8c",
    "APO-PSO": "#8464a8",
    "APO-wr-DE": "#1b5e46",
    "APO-wr-LS": "#7d2358",
    "APO-wr-PSO": "#4a3170",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def load_results(results_dir: Path):
    runs = read_csv(results_dir / "runs.csv")
    convergence = read_csv(results_dir / "convergence.csv")
    if not runs or not convergence:
        raise ValueError(f"no run or convergence records found in {results_dir}")

    run_fitness = {
        (row["algorithm"], row["trial"]): float(row["penalized_fitness"])
        for row in runs
    }
    traces: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in convergence:
        key = (row["algorithm"], row["trial"])
        traces[key].append(
            (int(row["evaluations"]), float(row["best_penalized_fitness"]))
        )

    if set(traces) != set(run_fitness):
        raise ValueError("runs.csv and convergence.csv contain different trials")
    for key, trace in traces.items():
        trace.sort()
        if not math.isclose(
            trace[-1][1], run_fitness[key], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"final convergence value does not match runs.csv for "
                f"{key[0]} trial {key[1]}; regenerate experiment outputs"
            )

    algorithms = list(dict.fromkeys(row["algorithm"] for row in runs))
    return runs, traces, algorithms


def plot_results(results_dir: Path, output_dir: Path) -> list[Path]:
    runs, traces, algorithms = load_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    final_fitness = np.asarray(
        [float(row["penalized_fitness"]) for row in runs]
    )
    evaluation_start = min(point[0] for trace in traces.values() for point in trace)
    evaluation_end = max(point[0] for trace in traces.values() for point in trace)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for algorithm in algorithms:
        algorithm_traces = [
            trace for (name, _), trace in traces.items() if name == algorithm
        ]
        evaluation_axes = np.asarray([point[0] for point in algorithm_traces[0]])
        if any(
            [point[0] for point in trace] != evaluation_axes.tolist()
            for trace in algorithm_traces[1:]
        ):
            raise ValueError(f"trials for {algorithm} use different evaluation steps")
        fitness = np.asarray([[point[1] for point in trace] for trace in algorithm_traces])
        median = np.median(fitness, axis=0)
        lower = np.percentile(fitness, 25, axis=0)
        upper = np.percentile(fitness, 75, axis=0)
        color = COLORS.get(algorithm)
        linestyle = "--" if "-wr-" in algorithm else "-"
        ax.plot(evaluation_axes, median, label=algorithm, color=color,
                linewidth=2, linestyle=linestyle)
        ax.fill_between(
            evaluation_axes, lower, upper, color=color, alpha=0.18, linewidth=0
        )
    ax.set(
        title="Spring design: convergence across independent trials",
        xlabel="Objective function evaluations",
        ylabel="Best-so-far penalized fitness (median, shaded IQR)",
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "convergence.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for algorithm in algorithms:
        algorithm_traces = [
            trace for (name, _), trace in traces.items() if name == algorithm
        ]
        evaluation_axes = np.asarray([point[0] for point in algorithm_traces[0]])
        fitness = np.asarray(
            [[point[1] for point in trace] for trace in algorithm_traces]
        )
        median = np.median(fitness, axis=0)
        lower = np.percentile(fitness, 25, axis=0)
        upper = np.percentile(fitness, 75, axis=0)
        color = COLORS.get(algorithm)
        linestyle = "--" if "-wr-" in algorithm else "-"
        ax.plot(evaluation_axes, median, label=algorithm, color=color,
                linewidth=2, linestyle=linestyle)
        ax.fill_between(
            evaluation_axes, lower, upper, color=color, alpha=0.18, linewidth=0
        )
    ax.set(
        title="Spring design: convergence detail",
        xlabel="Objective function evaluations",
        ylabel="Best-so-far penalized fitness (median, shaded IQR)",
        xlim=(evaluation_start, min(evaluation_end, 2000)),
        ylim=(
            max(np.finfo(float).tiny, 0.9 * float(final_fitness.min())),
            3 * float(np.median(final_fitness)),
        ),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "convergence_detail.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    fitness_by_algorithm = [
        [
            float(row["penalized_fitness"])
            for row in runs
            if row["algorithm"] == algorithm
        ]
        for algorithm in algorithms
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    boxplot = ax.boxplot(
        fitness_by_algorithm,
        tick_labels=algorithms,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333"},
        medianprops={"color": "#222222", "linewidth": 1.5},
    )
    for patch, algorithm in zip(boxplot["boxes"], algorithms):
        patch.set_facecolor(COLORS.get(algorithm, "#777777"))
        patch.set_alpha(0.72)
    ax.set(
        title="Distribution of final penalized fitness",
        xlabel="Algorithm",
        ylabel="Penalized fitness (lower is better)",
    )
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "final_fitness.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    feasibility = [
        np.mean(
            [
                int(row["feasible"])
                for row in runs
                if row["algorithm"] == algorithm
            ]
        )
        for algorithm in algorithms
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bars = ax.bar(
        algorithms,
        feasibility,
        color=[COLORS.get(name, "#777777") for name in algorithms],
        width=0.62,
    )
    for bar, rate in zip(bars, feasibility):
        ax.annotate(
            f"{rate:.0%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set(
        title="Feasible solutions across independent trials",
        xlabel="Algorithm",
        ylabel="Feasible-run rate",
        ylim=(0, 1.12),
    )
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "feasibility.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    return written

def plot_best_designs(results_dir: Path, output_dir: Path) -> list[Path]:
    """Save the best feasible spring per algorithm as a CSV and a figure."""
    runs = read_csv(results_dir / "runs.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for algorithm in dict.fromkeys(row["algorithm"] for row in runs):
        feasible = [
            row for row in runs
            if row["algorithm"] == algorithm and int(row["feasible"]) == 1
        ]
        if feasible:
            best_rows.append(min(feasible, key=lambda r: float(r["raw_objective"])))
    if not best_rows:
        raise ValueError("no feasible runs found")

    csv_path = results_dir / "best_designs.csv"
    fields = ["algorithm", "trial", "seed", "x1", "x2", "x3",
              "raw_objective", "penalized_fitness", "total_constraint_violation"]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in best_rows)

    # Sort best (lowest weight) first so the plot reads as a ranking.
    best_rows.sort(key=lambda r: float(r["raw_objective"]))
    names = [r["algorithm"] for r in best_rows]
    weights = np.array([float(r["raw_objective"]) for r in best_rows])

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ypos = np.arange(len(names))
    ax.scatter(weights, ypos, s=90,
               color=[COLORS.get(n, "#777777") for n in names], zorder=3)
    for y, row, weight in zip(ypos, best_rows, weights):
        ax.annotate(
            f"{weight:.8f}   (x1={float(row['x1']):.5f}, "
            f"x2={float(row['x2']):.5f}, x3={float(row['x3']):.4f})",
            (weight, y), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=8,
        )
    ax.axvline(0.012665, color="#555555", linestyle="--", linewidth=1,
               label="literature best ≈ 0.012665")
    span = max(weights.max() - weights.min(), 1e-6)
    ax.set_xlim(weights.min() - 0.5 * span, weights.max() + 4 * span)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set(
        title="Best feasible spring found by each algorithm",
        xlabel="Spring weight (raw objective, lower is better)",
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    path = output_dir / "best_designs.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return [csv_path, path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_all"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_all") / "figures"
    )
    args = parser.parse_args()
    for path in plot_results(args.results_dir, args.output_dir):
        print(f"Saved {path}")
    for path in plot_best_designs(args.results_dir, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
