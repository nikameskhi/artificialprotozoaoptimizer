# =============================================================================
#  ALGORITHM REFERENCE
#  ---------------------------------------------------------------------------
#  Algorithm: Artificial Protozoa Optimizer (APO)
#  Author:    Wang Xiaopeng, et al.
#  Journal:   Knowledge-Based Systems (2024)
#  DOI:       https://doi.org/10.1016/j.knosys.2024.111737
# =============================================================================

import numpy as np
from copy import deepcopy

# =============================================================================
# APO Core Algorithm
# =============================================================================
def _apo_optimizer(
    func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max,
    *, rank_weights=False, de_fraction=0.0, de_scale=0.5, de_crossover=0.9,
    local_search_fraction=0.0, local_search_scale=0.1, pso_fraction=0.0,
):
    if not 0.0 <= de_fraction <= 1.0:
        raise ValueError("de_fraction must be between 0 and 1")
    if not 0.0 <= local_search_fraction <= 1.0:
        raise ValueError("local_search_fraction must be between 0 and 1")
    if not 0.0 <= pso_fraction <= 1.0:
        raise ValueError("pso_fraction must be between 0 and 1")
    if de_fraction > 0 and pop_size < 4:
        raise ValueError("population size must be at least 4 when DE is enabled")
    if not 0.0 <= de_crossover <= 1.0:
        raise ValueError("de_crossover must be between 0 and 1")
    if de_scale < 0:
        raise ValueError("de_scale must be non-negative")
    if local_search_scale < 0:
        raise ValueError("local_search_scale must be non-negative")

    Xmin, Xmax = np.array(lb), np.array(ub)

    # Initialization
    protozoa = Xmin + np.random.rand(pop_size, dim) * (Xmax - Xmin)
    protozoa_fit = np.array([func.evaluate(ind) for ind in protozoa])
    velocities = np.zeros_like(protozoa)
    personal_best = protozoa.copy()
    personal_best_fit = protozoa_fit.copy()

    best_id = np.argmin(protozoa_fit)
    best_fit = protozoa_fit[best_id]
    best_protozoa = deepcopy(protozoa[best_id])
    trace = [best_fit]

    for it in range(1, iter_max):
        # 1. Ranking/Sorting: protozoa population
        idx_sort = np.argsort(protozoa_fit)
        protozoa = protozoa[idx_sort]
        protozoa_fit = protozoa_fit[idx_sort]
        velocities = velocities[idx_sort]
        personal_best = personal_best[idx_sort]
        personal_best_fit = personal_best_fit[idx_sort]

        new_protozoa = np.zeros((pop_size, dim))
        pf = pf_max * np.random.rand()
        # Randomly select indices for dormancy or reproduction
        ri = np.random.choice(range(pop_size), int(np.ceil(pop_size * pf)), replace=False)

        for i in range(pop_size):
            if i in ri:
                # --- A. Dormancy and Reproduction Forms ---
                pdr = 0.5 * (1 + np.cos((1 - (i+1) / pop_size) * np.pi))
                if np.random.rand() < pdr:
                    # Dormancy: Random re-initialization
                    new_protozoa[i] = Xmin + np.random.rand(dim) * (Xmax - Xmin)
                else:
                    # Reproduction: Local perturbation
                    flag = 1 if np.random.rand() < 0.5 else -1
                    mr = np.zeros(dim)
                    # Mr is a mapping vector in reproduction
                    mr[np.random.choice(range(dim), int(np.ceil(np.random.rand()*dim)), replace=False)] = 1
                    new_protozoa[i] = protozoa[i] + flag * np.random.rand() * (Xmin + np.random.rand(dim) * (Xmax - Xmin)) * mr
            else:
                # --- B. Foraging Form ---
                f_factor = np.random.rand() * (1 + np.cos(it / iter_max * np.pi))
                mf = np.zeros(dim)
                # Mf is a mapping vector in foraging
                mf[np.random.choice(range(dim), int(np.ceil(dim * (i+1) / pop_size)), replace=False)] = 1
                pah = 0.5 * (1 + np.cos(it / iter_max * np.pi))
                epn = np.zeros((np_pairs, dim))

                if np.random.rand() < pah:
                    # Autotroph
                    j = np.random.randint(0, pop_size)
                    for k in range(1, np_pairs + 1):
                        if i == 0: km, kp = i, i + np.random.randint(1, pop_size - i)
                        elif i == pop_size - 1: km, kp = np.random.randint(0, pop_size - 1), i
                        else: km, kp = np.random.randint(0, i), i + np.random.randint(1, pop_size - i)
                        # Weight factor for autotroph form
                        if rank_weights:
                            wa = np.exp(-(1 - (km + 1) / (kp + 1)))
                        else:
                            wa = np.exp(-abs(protozoa_fit[km] / (protozoa_fit[kp] + np.finfo(float).eps)))
                        # Replace fitness-based weighting with ranking. Rank starts from 1.
                        # wa = np.exp(-(1 - (km + 1) / (kp + 1)))
                        epn[k-1] = wa * (protozoa[km] - protozoa[kp])
                    new_protozoa[i] = protozoa[i] + f_factor * (protozoa[j] - protozoa[i] + (1/np_pairs) * np.sum(epn, axis=0)) * mf
                else:
                    # Heterotroph
                    for k in range(1, np_pairs + 1):
                        imk, ipk = max(0, i - k), min(pop_size - 1, i + k)
                        # Weight factor for heterotroph form
                        if rank_weights:
                            wh = np.exp(-(1 - (imk + 1) / (ipk + 1)))
                        else:
                            wh = np.exp(-abs(protozoa_fit[imk] / (protozoa_fit[ipk] + np.finfo(float).eps)))
                        # Replace fitness-based weighting with ranking. Rank starts from 1.
                        # wh = np.exp(-(1 - (imk + 1) / (ipk + 1)))
                        epn[k-1] = wh * (protozoa[imk] - protozoa[ipk])
                    flag = 1 if np.random.rand() < 0.5 else -1
                    x_near = (1 + flag * np.random.rand(dim) * (1 - it/iter_max)) * protozoa[i]
                    new_protozoa[i] = protozoa[i] + f_factor * (x_near - protozoa[i] + (1/np_pairs) * np.sum(epn, axis=0)) * mf

        if de_fraction > 0:
            de_count = int(np.ceil(pop_size * de_fraction))
            de_targets = np.random.choice(pop_size, de_count, replace=False)
            for target in de_targets:
                donors = np.random.choice(
                    np.delete(np.arange(pop_size), target), 3, replace=False
                )
                mutant = protozoa[donors[0]] + de_scale * (
                    protozoa[donors[1]] - protozoa[donors[2]]
                )
                crossover_mask = np.random.rand(dim) < de_crossover
                crossover_mask[np.random.randint(dim)] = True
                new_protozoa[target] = np.where(
                    crossover_mask, mutant, new_protozoa[target]
                )

        if local_search_fraction > 0:
            local_count = int(np.ceil(pop_size * local_search_fraction))
            local_targets = range(pop_size - local_count, pop_size)
            radius = local_search_scale * (1 - it / iter_max)
            for target in local_targets:
                new_protozoa[target] = best_protozoa + np.random.normal(
                    0, radius * (Xmax - Xmin), dim
                )

        if pso_fraction > 0:
            pso_count = int(np.ceil(pop_size * pso_fraction))
            pso_targets = np.random.choice(pop_size, pso_count, replace=False)
            inertia = 0.9 - 0.5 * (it / iter_max)
            velocity_limit = 0.2 * (Xmax - Xmin)
            for target in pso_targets:
                velocities[target] = (
                    inertia * velocities[target]
                    + 1.5 * np.random.rand(dim)
                    * (personal_best[target] - protozoa[target])
                    + 1.5 * np.random.rand(dim)
                    * (best_protozoa - protozoa[target])
                )
                velocities[target] = np.clip(
                    velocities[target], -velocity_limit, velocity_limit
                )
                new_protozoa[target] = protozoa[target] + velocities[target]

        # Boundary control and Greedy selection
        new_protozoa = np.clip(new_protozoa, Xmin, Xmax)
        for i in range(pop_size):
            nf = func.evaluate(new_protozoa[i])
            if nf < protozoa_fit[i]:
                protozoa_fit[i], protozoa[i] = nf, new_protozoa[i]
            if protozoa_fit[i] < personal_best_fit[i]:
                personal_best_fit[i] = protozoa_fit[i]
                personal_best[i] = protozoa[i]

        # Update current best
        c_best = np.argmin(protozoa_fit)
        best_fit = protozoa_fit[c_best]
        best_protozoa = deepcopy(protozoa[c_best])

        trace.append(best_fit)

    return trace, best_protozoa, best_fit
def APO_orig(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max):
    return _apo_optimizer(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max)
