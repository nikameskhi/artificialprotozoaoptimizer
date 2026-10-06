"""Constrained tension/compression spring design benchmark."""

import numpy as np


LOWER_BOUNDS = np.array([0.05, 0.25, 2.0])
UPPER_BOUNDS = np.array([2.0, 1.3, 15.0])


class SpringDesign:
    """Spring design objective with the standard four inequality constraints."""

    penalty = 1e6

    def components(self, x):
        wire_diameter, mean_coil_diameter, active_coils = np.asarray(x, dtype=float)
        objective = (active_coils + 2) * mean_coil_diameter * wire_diameter**2
        constraints = np.array(
            [
                1 - (
                    mean_coil_diameter**3 * active_coils
                    / (71785 * wire_diameter**4)
                ),
                (
                    (4 * mean_coil_diameter**2 - wire_diameter * mean_coil_diameter)
                    / (
                        12566
                        * (
                            mean_coil_diameter * wire_diameter**3
                            - wire_diameter**4
                        )
                    )
                    + 1 / (5108 * wire_diameter**2)
                    - 1
                ),
                1 - (
                    140.45 * wire_diameter
                    / (mean_coil_diameter**2 * active_coils)
                ),
                (wire_diameter + mean_coil_diameter) / 1.5 - 1,
            ],
            dtype=float,
        )
        violations = np.maximum(constraints, 0.0)
        violation_total = float(np.sum(violations))
        penalized = float(objective + self.penalty * np.sum(violations**2))
        return penalized, float(objective), violation_total

    def evaluate(self, x):
        return self.components(x)[0]


def is_feasible(x, tolerance=1e-8):
    _, _, total_violation = SpringDesign().components(x)
    return total_violation <= tolerance
