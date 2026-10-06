"""Artificial Protozoa Optimizer hybridized with Differential Evolution."""

from apo import _apo_optimizer


def APO_DE(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max):
    """Replace 20% of each generation's candidates with DE/rand/1/bin."""
    return _apo_optimizer(
        func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max,
        de_fraction=0.2, de_scale=0.5, de_crossover=0.9,
    )
