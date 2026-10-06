"""Rank-weighted Artificial Protozoa Optimizer variant."""

from apo import _apo_optimizer


def APO_wr(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max):
    """Run APO using rank-based autotroph and heterotroph weights."""
    return _apo_optimizer(
        func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max,
        rank_weights=True,
    )
