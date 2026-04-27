import random


def _make_trial(ind, mutant_list, cr, creator):
    """
    Aplica crossover binomial entre o indivíduo atual e o vetor mutante.
    """
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))

    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])

    del trial.fitness.values
    return trial


def de_rand_1_bin(ind, population, f, cr, creator, toolbox):
    """
    DE/rand/1/bin
    v = a + F * (b - c)
    """
    candidates = [x for x in population if x is not ind]

    if len(candidates) < 3:
        return ind

    a, b, c = toolbox.rnd_selection(candidates, 3)

    mutant_list = [
        a_i + f * (b_i - c_i)
        for a_i, b_i, c_i in zip(a, b, c)
    ]

    return _make_trial(ind, mutant_list, cr, creator)


def de_rand_2_bin(ind, population, f, cr, creator, toolbox):
    """
    DE/rand/2/bin
    v = a + F * (b - c) + F * (d - e)
    """
    candidates = [x for x in population if x is not ind]

    if len(candidates) < 5:
        return ind

    a, b, c, d, e = toolbox.rnd_selection(candidates, 5)

    mutant_list = [
        a_i + f * (b_i - c_i) + f * (d_i - e_i)
        for a_i, b_i, c_i, d_i, e_i in zip(a, b, c, d, e)
    ]

    return _make_trial(ind, mutant_list, cr, creator)


def de_best_1_bin(ind, population, best, f, cr, creator, toolbox):
    """
    DE/best/1/bin
    v = best + F * (a - b)
    """
    candidates = [
        x for x in population
        if x is not ind and x is not best
    ]

    if len(candidates) < 2:
        return ind

    a, b = toolbox.rnd_selection(candidates, 2)

    mutant_list = [
        best_i + f * (a_i - b_i)
        for best_i, a_i, b_i in zip(best, a, b)
    ]

    return _make_trial(ind, mutant_list, cr, creator)


def de_rand_to_best_2_bin(ind, population, best, f, cr, creator, toolbox):
    """
    DE/rand-to-best/2/bin

    Variante implementada:
    v = best + F * (a - b) + F * (c - d)

    Obs.: é uma estratégia mais agressiva/explotativa.
    """
    candidates = [
        x for x in population
        if x is not ind and x is not best
    ]

    if len(candidates) < 4:
        return ind

    a, b, c, d = toolbox.rnd_selection(candidates, 4)

    mutant_list = [
        best_i + f * (a_i - b_i) + f * (c_i - d_i)
        for best_i, a_i, b_i, c_i, d_i in zip(best, a, b, c, d)
    ]

    return _make_trial(ind, mutant_list, cr, creator)


def de_current_to_rand_1(ind, population, f, cr, creator, toolbox):
    """
    DE/current-to-rand/1/bin

    Variante com crossover binomial:
    v = current + F * (x_rand - current) + F * (a - b)
    """
    candidates = [x for x in population if x is not ind]

    if len(candidates) < 3:
        return ind

    x_rand, a, b = toolbox.rnd_selection(candidates, 3)

    mutant_list = [
        ind_i + f * (x_rand_i - ind_i) + f * (a_i - b_i)
        for ind_i, x_rand_i, a_i, b_i in zip(ind, x_rand, a, b)
    ]

    return _make_trial(ind, mutant_list, cr, creator)


def de_current_to_pbest_1(ind, population, f, cr, creator, toolbox, p=0.1):
    """
    DE/current-to-pbest/1/bin, inspirado no JADE.

    v = current + F * (p_best - current) + F * (a - b)
    """
    if len(population) < 4:
        return ind

    num_p_best = max(1, int(len(population) * p))

    top_p_individuals = sorted(
        population,
        key=lambda x: x.fitness.values[0]
    )[:num_p_best]

    p_best = random.choice(top_p_individuals)

    candidates = [
        x for x in population
        if x is not ind and x is not p_best
    ]

    if len(candidates) < 2:
        return ind

    a, b = toolbox.rnd_selection(candidates, 2)

    mutant_list = [
        ind_i + f * (pbest_i - ind_i) + f * (a_i - b_i)
        for ind_i, pbest_i, a_i, b_i in zip(ind, p_best, a, b)
    ]

    return _make_trial(ind, mutant_list, cr, creator)