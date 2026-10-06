"""Artificial Protozoa Optimizer hybridized with Particle Swarm Optimization."""

from apo import _apo_optimizer


def APO_PSO(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max):
    """Use PSO updates for 10% of candidates in each generation."""
    return _apo_optimizer(
        func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max,
        pso_fraction=0.1,
    )
