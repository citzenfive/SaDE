import numpy as np
import random


### Funções de Mutação para Evolução Diferencial (DEAP) ###

def de_rand_1_bin(ind, population, f, cr, creator, toolbox):
    """
    Estratégia DE/rand/1/bin.
    Cria um vetor mutante a partir de 3 indivíduos aleatórios e aplica o crossover binomial.
    """
    # 1. Checagem de segurança
    if len(population) < 4:
        return ind

    # 2. Seleção de vetores e criação do mutante
    a, b, c = toolbox.rnd_selection(population, 3)
    mutant_list = [a_i + f * (b_i - c_i) for a_i, b_i, c_i in zip(a, b, c)]
    
    # 3. Crossover Binomial
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))
    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])
            
    del trial.fitness.values
    return trial

# ---

def de_rand_to_best_2_bin(ind, population, best, f, cr, creator, toolbox):
    """
    Estratégia DE/rand-to-best/2/bin.
    Usa o melhor indivíduo e 2 vetores de diferença, seguido de crossover binomial.
    """
    # 1. Checagem de segurança
    if len(population) < 5:
        return ind

    # 2. Seleção de vetores e criação do mutante
    a, b, c, d = random.sample([x for x in population if x != ind and x != best], 4)
    mutant_list = [best_i + f * (a_i - b_i) + f * (c_i - d_i) for best_i, a_i, b_i, c_i, d_i in zip(best, a, b, c, d)]
    
    # 3. Crossover Binomial
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))
    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])
            
    del trial.fitness.values
    return trial

# ---

def de_rand_2_bin(ind, population, f, cr, creator, toolbox):
    """
    Estratégia DE/rand/2/bin.
    Usa 2 vetores de diferença para criar o mutante, seguido de crossover binomial.
    """
    # 1. Checagem de segurança
    if len(population) < 6:
        return ind

    # 2. Seleção de vetores e criação do mutante
    a, b, c, d, e = random.sample([x for x in population if x != ind], 5)
    mutant_list = [a_i + f * (b_i - c_i) + f * (d_i - e_i) for a_i, b_i, c_i, d_i, e_i in zip(a, b, c, d, e)]
    
    # 3. Crossover Binomial
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))
    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])
            
    del trial.fitness.values
    return trial

# ---

def de_current_to_rand_1(ind, population, f, cr, creator, toolbox):
    """
    Estratégia DE/current-to-rand/1.
    Usa o indivíduo atual ('current') e um aleatório ('rand'), seguido de crossover.
    """
    # 1. Checagem de segurança
    if len(population) < 4:
        return ind

    # 2. Seleção de vetores e criação do mutante
    x_rand, a, b = random.sample([x for x in population if x != ind], 3)
    mutant_list = [ind_i + f * (x_rand_i - ind_i) + f * (a_i - b_i) for ind_i, x_rand_i, a_i, b_i in zip(ind, x_rand, a, b)]
    
    # 3. Crossover Binomial
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))
    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])
            
    del trial.fitness.values
    return trial