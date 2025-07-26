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

# ==============================================================================
# ESTRATÉGIA 1: DE/best/1/bin (Focada em Explotação / "Agressiva")
# ==============================================================================
def de_best_1_bin(ind, population, best, f, cr, creator, toolbox):
    """
    Estratégia de mutação DE/best/1/bin.
    
    Usa o melhor indivíduo ('best') como base para guiar a busca,
    tornando-a mais "gananciosa" e focada em explorar a melhor região encontrada.
    É excelente para o refinamento de soluções.
    """
    # --- 1. Checagem de Segurança ---
    # A fórmula precisa do 'ind', 'best', 'a', e 'b'.
    # A população precisa ter pelo menos 4 indivíduos para a seleção segura.
    if len(population) < 4:
        return ind

    # --- 2. Seleção dos Vetores ---
    # Cria uma lista de candidatos para 'a' e 'b' que não sejam o indivíduo
    # atual ('ind') nem o melhor da população ('best').
    candidates = [x for x in population if x != ind and x != best]
    if len(candidates) < 2:
        return ind # Retorna o original se não houver candidatos suficientes

    a, b = toolbox.rnd_selection(candidates, 2)
    
    # --- 3. Criação do Vetor Mutante ---
    mutant_list = [best_i + f * (a_i - b_i) for best_i, a_i, b_i in zip(best, a, b)]
    
    # --- 4. Crossover Binomial ---
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))
    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])
            
    # --- 5. Finalização ---
    # Invalida o fitness do novo indivíduo para garantir que ele seja reavaliado.
    del trial.fitness.values
    return trial


# ==============================================================================
# ESTRATÉGIA 2: DE/current-to-p-best/1 (Híbrida Avançada)
# ==============================================================================
def de_current_to_pbest_1(ind, population, f, cr, creator, toolbox, p=0.1):
    """
    Estratégia de mutação DE/current-to-p-best/1, inspirada no algoritmo JADE.

    Perturba o indivíduo atual ('current') em direção a uma solução de elite
    ('p-best') escolhida aleatoriamente do topo 'p'% da população. Oferece um
    excelente equilíbrio entre exploração e explotação.
    """
    # --- 1. Checagem de Segurança ---
    if len(population) < 4:
        return ind
    
    # --- 2. Seleção dos Vetores ---
    
    # Passo 2a: Encontrar o indivíduo 'p_best'
    # Determina o número de indivíduos no conjunto de elite (pelo menos 1).
    num_p_best = max(1, int(len(population) * p))
    
    # Ordena a população pelo fitness (assumindo minimização) e pega os melhores.
    top_p_individuals = sorted(population, key=lambda ind: ind.fitness.values[0])[:num_p_best]
    
    # Escolhe um 'p_best' aleatoriamente desse conjunto de elite.
    p_best = random.choice(top_p_individuals)

    # Passo 2b: Selecionar os vetores 'a' e 'b'
    # Eles devem ser diferentes do 'ind' e do 'p_best' recém-escolhido.
    candidates = [x for x in population if x != ind and x != p_best]
    if len(candidates) < 2:
        return ind
        
    a, b = toolbox.rnd_selection(candidates, 2)
    
    # --- 3. Criação do Vetor Mutante ---
    # Fórmula: v = x_atual + F * (x_pbest - x_atual) + F * (a - b)
    mutant_list = [ind_i + f * (pbest_i - ind_i) + f * (a_i - b_i) 
                   for ind_i, pbest_i, a_i, b_i in zip(ind, p_best, a, b)]
    
    # --- 4. Crossover Binomial ---
    trial = creator.Individual()
    rand_j = random.randrange(len(ind))
    for i in range(len(ind)):
        if random.random() < cr or i == rand_j:
            trial.append(mutant_list[i])
        else:
            trial.append(ind[i])
            
    # --- 5. Finalização ---
    del trial.fitness.values
    return trial