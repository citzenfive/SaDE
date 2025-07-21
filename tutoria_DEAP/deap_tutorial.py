# Minha implementação do SaDE usando o toolbox DEAP como base

import numpy as np
import pandas as pd
import statsmodels as stt
import functools

from numpy.random import Generator, MT19937, SeedSequence

from scipy.stats import qmc

from deap import base
from deap import creator
from deap import gp
from deap import tools

from evaluation import *
from mut_cross import *

# Configuração da execução do algoritmo
config_seed = 1234
pop_size = 10
gen_number = 2000
parallel_computing = False
bounds = [(1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1)]
l_bounds = [i[0] for i in bounds]
u_bounds = [i[1] for i in bounds]
gmax_non_imp = 0.15*gen_number
F = 0.5
CR = 0.15

# Configuração do gerador de números aleatórios
prng = Generator(MT19937(seed=config_seed))


# Configuração do gerador da população inicial
sampler = qmc.Sobol(d=len(bounds), scramble=True, rng=prng)
m = np.log2(pop_size)
m = int(np.ceil(m))
sample = sampler.random_base2(m=m)
qmc.scale(sample=sample, l_bounds=l_bounds, u_bounds=u_bounds) # Garante que as minhas amostras estejam dentro do intervalo necessário

# Boas estratégias:
'''
    -> DE/rand/1/bin        
    -> DE/rand-to-best/2/bin
    -> DE/rand/2/bin        
    -> DE/current-to-rand/1 
'''


# Criando classes da toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
CONFIGURED_CREATOR = creator

# Criando a população a parir de uma população já criada
pop_SaDE = []

for smp in sample:
    pop_SaDE.append(creator.Individual(smp.tolist()))

# Realizando a evaluation do fitness
for pos, ind in enumerate(pop_SaDE):
    ind.fitness.values = evaluate(ind)

print(pop_SaDE)
print("\n\n")
test = pop_SaDE.copy()
print(test)
print("\n\n")
print(pop_SaDE[0].fitness.values)
print(test[0].fitness.values)
print("\n\n")

# Usando o toolbox 
# Registrando o toolbox
toolbox = base.Toolbox()

toolbox.register("evaluate", evaluate) # aqui eu estou registrando a função que faz a avaliação do meu individuo
toolbox.register("rnd_selection", tools.selRandom)
toolbox.register("select_next", tools.selTournament)

# Realizando a mutação. Como a operação de mutação apenas realiza mutação, preciso primeiro fazer uma cópia do individuo
# se necessário for mante-lo. Após feita essa cópia, faço a mutação gaussiana no mesmo 
# Essa função já está registrada na biblioteca
mutant = toolbox.clone(pop_SaDE[1])
ind2, = tools.mutGaussian(mutant, mu=0.5, sigma=0.3, indpb=0.2)

del mutant.fitness.values # o fitness value deve ser excluido pois nao é do individuo mutante e sim do seu originário

print(pop_SaDE[1])
print(mutant)
print(ind2)


# No meu arquivo de mutation.py eu programei algumas mutações que já realizam crossover diferentes
# Sendo assim, tenho que registrá-las

toolbox.register("rand_1_bin", de_rand_1_bin)
toolbox.register("rand_to_best_2_bin", de_rand_to_best_2_bin)
toolbox.register("de_rand_2_bin", de_rand_2_bin)
toolbox.register("de_current_to_rand_1", de_current_to_rand_1)

idv3 = toolbox.rand_1_bin(pop_SaDE[0].copy(), pop_SaDE, f=F, cr=CR, creator=CONFIGURED_CREATOR, toolbox=toolbox)
print(idv3)

print(idv3.fitness.valid)


# Selectionar individuos que ainda não foram avaliados
invalid_idv = [idv for idv in pop_SaDE if not idv.fitness.valid]
print(invalid_idv)

print("\n\n\n")

bounds = [(0.10, 0.20), (1.5, 1.9), (0.01, 0.15), (0, 0.13), (0.1, 0.25), (0.5, 0.6)]
l_bounds = [i[0] for i in bounds]
u_bounds = [i[1] for i in bounds]

def bound_maker(offspring):
    for pos, par in enumerate(offspring):
        print(par)
        par = (l_bounds[pos]) + par * (u_bounds[pos] - l_bounds[pos])
        print(par)
        offspring[pos] = par

idv0 = (creator.Individual([0, 1, -3, -4, 9, -1]))

print(idv3)
print("\n\n\n")
bound_maker(offspring=idv0)
print("\n\n\n")
print("\n\n\n")
print("\n\n\n")
print("\n\n\n")
print(idv3)
a = idv3[0]
print(idv3[0])