"""Artificial Protozoa Optimizer with a shrinking local-search neighborhood."""

from apo import _apo_optimizer


def APO_LS(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max):
    """Use 10% of each generation for local search around the best solution."""
    return _apo_optimizer(
        func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max,
        local_search_fraction=0.1,
        local_search_scale=0.1,
    )
