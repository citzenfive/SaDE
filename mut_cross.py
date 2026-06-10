"""Mutation and crossover strategies used by the SaDE implementation.

All random operations receive the same NumPy ``Generator`` used by SaDE,
which makes complete runs reproducible from a single seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import Generator


Individual = Any


def _sample_distinct(
    population: Sequence[Individual],
    count: int,
    rng: Generator,
    *excluded: Individual,
) -> list[Individual] | None:
    """Sample distinct individuals while excluding objects by identity."""
    excluded_ids = {id(ind) for ind in excluded if ind is not None}
    valid_indices = [
        index for index, candidate in enumerate(population)
        if id(candidate) not in excluded_ids
    ]

    if len(valid_indices) < count:
        return None

    selected = rng.choice(valid_indices, size=count, replace=False)
    return [population[int(index)] for index in np.atleast_1d(selected)]


def _make_trial(
    target: Individual,
    mutant: np.ndarray,
    cr: float,
    creator: Any,
    rng: Generator,
) -> Individual:
    """Apply vectorized binomial crossover to a mutant vector."""
    target_array = np.asarray(target, dtype=float)
    crossover_mask = rng.random(target_array.size) < cr
    crossover_mask[int(rng.integers(target_array.size))] = True

    trial_values = np.where(crossover_mask, mutant, target_array)
    return creator.Individual(trial_values.tolist())


def de_rand_1_bin(ind, population, f, cr, creator, rng, **_):
    """DE/rand/1/bin: ``v = a + F (b - c)``."""
    selected = _sample_distinct(population, 3, rng, ind)
    if selected is None:
        return None

    a, b, c = map(lambda x: np.asarray(x, dtype=float), selected)
    mutant = a + f * (b - c)
    return _make_trial(ind, mutant, cr, creator, rng)


def de_rand_2_bin(ind, population, f, cr, creator, rng, **_):
    """DE/rand/2/bin: ``v = a + F (b - c) + F (d - e)``."""
    selected = _sample_distinct(population, 5, rng, ind)
    if selected is None:
        return None

    a, b, c, d, e = map(lambda x: np.asarray(x, dtype=float), selected)
    mutant = a + f * (b - c) + f * (d - e)
    return _make_trial(ind, mutant, cr, creator, rng)


def de_best_1_bin(ind, population, best, f, cr, creator, rng, **_):
    """DE/best/1/bin: ``v = best + F (a - b)``."""
    selected = _sample_distinct(population, 2, rng, ind, best)
    if selected is None:
        return None

    a, b = map(lambda x: np.asarray(x, dtype=float), selected)
    mutant = np.asarray(best, dtype=float) + f * (a - b)
    return _make_trial(ind, mutant, cr, creator, rng)


def de_rand_to_best_2_bin(ind, population, best, f, cr, creator, rng, **_):
    """DE/rand-to-best/2/bin.

    ``v = a + F (best - a) + F (b - c) + F (d - e)``

    The previous implementation used ``best + F(a-b) + F(c-d)``, which is
    DE/best/2 rather than DE/rand-to-best/2.
    """
    selected = _sample_distinct(population, 5, rng, ind, best)
    if selected is None:
        return None

    a, b, c, d, e = map(lambda x: np.asarray(x, dtype=float), selected)
    best_array = np.asarray(best, dtype=float)
    mutant = a + f * (best_array - a) + f * (b - c) + f * (d - e)
    return _make_trial(ind, mutant, cr, creator, rng)


def de_current_to_rand_1(ind, population, f, cr, creator, rng, **_):
    """Current-to-rand/1 with binomial crossover.

    ``v = current + F (x_rand - current) + F (a - b)``
    """
    selected = _sample_distinct(population, 3, rng, ind)
    if selected is None:
        return None

    x_rand, a, b = map(lambda x: np.asarray(x, dtype=float), selected)
    current = np.asarray(ind, dtype=float)
    mutant = current + f * (x_rand - current) + f * (a - b)
    return _make_trial(ind, mutant, cr, creator, rng)


def de_current_to_pbest_1(
    ind,
    population,
    f,
    cr,
    creator,
    rng,
    pbest_pool=None,
    p=0.1,
    **_,
):
    """DE/current-to-pbest/1/bin inspired by JADE.

    ``v = current + F (pbest - current) + F (a - b)``

    ``pbest_pool`` should be precomputed once per generation by SaDE, avoiding
    one complete population sort for every individual.
    """
    if pbest_pool is None:
        pbest_count = max(2, int(np.ceil(len(population) * p)))
        pbest_pool = sorted(
            population, key=lambda individual: individual.fitness.values[0]
        )[:pbest_count]

    valid_pbest = [candidate for candidate in pbest_pool if candidate is not ind]
    if not valid_pbest:
        return None

    pbest = valid_pbest[int(rng.integers(len(valid_pbest)))]
    selected = _sample_distinct(population, 2, rng, ind, pbest)
    if selected is None:
        return None

    a, b = map(lambda x: np.asarray(x, dtype=float), selected)
    current = np.asarray(ind, dtype=float)
    pbest_array = np.asarray(pbest, dtype=float)
    mutant = current + f * (pbest_array - current) + f * (a - b)
    return _make_trial(ind, mutant, cr, creator, rng)