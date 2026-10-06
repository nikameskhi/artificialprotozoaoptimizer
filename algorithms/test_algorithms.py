import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apo import APO_orig
from apo_de import APO_DE
from apo_ls import APO_LS
from apo_pso import APO_PSO
from apo_wr import APO_wr
from apo_wr_de import APO_wr_DE
from apo_wr_ls import APO_wr_LS
from apo_wr_pso import APO_wr_PSO
from run_experiments import paired_statistics, summarize
from spring_design import LOWER_BOUNDS, UPPER_BOUNDS, SpringDesign
from visualize_results import load_results, plot_results


class Sphere:
    def evaluate(self, x):
        return float(np.dot(x, x))


class CountingSphere(Sphere):
    def __init__(self):
        self.evaluations = 0

    def evaluate(self, x):
        self.evaluations += 1
        return super().evaluate(x)


class AlgorithmTests(unittest.TestCase):
    def test_spring_design_known_feasible_design(self):
        objective = SpringDesign()
        penalized, raw, violation = objective.components(
            np.array([0.05255176574249507, 0.3776360283308216, 10.173440198548397])
        )
        self.assertGreater(raw, 0)
        self.assertEqual(penalized, raw)
        self.assertEqual(violation, 0)
        self.assertEqual(objective.evaluate(LOWER_BOUNDS), objective.components(LOWER_BOUNDS)[0])
        self.assertTrue(np.all(LOWER_BOUNDS < UPPER_BOUNDS))

    def test_variants_are_reproducible_bounded_and_monotonic(self):
        algorithms = (
            APO_orig, APO_wr, APO_DE, APO_LS, APO_PSO,
            APO_wr_DE, APO_wr_LS, APO_wr_PSO,
        )
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm.__name__):
                np.random.seed(71)
                first = algorithm(Sphere(), 3, 8, 5, [-2] * 3, [2] * 3, 1, 0.1)
                np.random.seed(71)
                second = algorithm(Sphere(), 3, 8, 5, [-2] * 3, [2] * 3, 1, 0.1)

                trace, position, fitness = first
                np.testing.assert_array_equal(trace, second[0])
                np.testing.assert_array_equal(position, second[1])
                self.assertEqual(fitness, second[2])
                self.assertEqual(len(trace), 5)
                self.assertTrue(np.all(np.diff(trace) <= 0))
                self.assertTrue(np.all(position >= -2))
                self.assertTrue(np.all(position <= 2))
                self.assertEqual(fitness, trace[-1])

    def test_local_search_uses_existing_evaluation_budget(self):
        for algorithm in (APO_LS, APO_wr_LS):
            with self.subTest(algorithm=algorithm.__name__):
                objective = CountingSphere()
                algorithm(objective, 3, 8, 5, [-2] * 3, [2] * 3, 1, 0.1)
                self.assertEqual(objective.evaluations, 8 * 5)

    def test_pso_hybrid_uses_existing_evaluation_budget(self):
        for algorithm in (APO_PSO, APO_wr_PSO):
            with self.subTest(algorithm=algorithm.__name__):
                objective = CountingSphere()
                algorithm(objective, 3, 8, 5, [-2] * 3, [2] * 3, 1, 0.1)
                self.assertEqual(objective.evaluations, 8 * 5)

    def test_de_requires_at_least_four_candidates(self):
        for algorithm in (APO_DE, APO_wr_DE):
            with self.subTest(algorithm=algorithm.__name__):
                with self.assertRaises(ValueError):
                    algorithm(Sphere(), 3, 3, 2, [-2] * 3, [2] * 3, 1, 0.1)




class VisualizationTests(unittest.TestCase):
    def test_creates_all_plots_from_consistent_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_dir = root / "results"
            figures_dir = root / "figures"
            results_dir.mkdir()

            run_rows = []
            convergence_rows = []
            for algorithm, fitness in (("APO", 1.0), ("APO-wr", 0.8)):
                run_rows.append(
                    [algorithm, 1, 2022, fitness, fitness, 0, 1, 0.1, 0.2, 0.3]
                )
                convergence_rows.extend(
                    [[algorithm, 1, 2022, 4, fitness + 0.5],
                     [algorithm, 1, 2022, 8, fitness]]
                )

            with (results_dir / "runs.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "algorithm",
                        "trial",
                        "seed",
                        "penalized_fitness",
                        "raw_objective",
                        "total_constraint_violation",
                        "feasible",
                        "x1",
                        "x2",
                        "x3",
                    ]
                )
                writer.writerows(run_rows)
            with (results_dir / "convergence.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "algorithm",
                        "trial",
                        "seed",
                        "evaluations",
                        "best_penalized_fitness",
                    ]
                )
                writer.writerows(convergence_rows)

            self.assertEqual(len(load_results(results_dir)[2]), 2)
            paths = plot_results(results_dir, figures_dir)
            self.assertEqual(len(paths), 4)
            self.assertTrue(
                all(path.is_file() and path.stat().st_size > 0 for path in paths)
            )

    def test_rejects_run_trace_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            (results_dir / "runs.csv").write_text(
                "algorithm,trial,penalized_fitness\nAPO,1,1.0\n",
                encoding="utf-8",
            )
            (results_dir / "convergence.csv").write_text(
                "algorithm,trial,evaluations,best_penalized_fitness\nAPO,1,10,2.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_results(results_dir)


if __name__ == "__main__":
    unittest.main()
