import numpy as np
import pandas as pd
import statsmodels as stt
import time

from numpy.random import Generator, MT19937, SeedSequence

from scipy.stats import qmc

from deap import base
from deap import creator
from deap import gp
from deap import tools

from evaluation import *
from mut_cross import *

class SaDE:
    def __init__(self, EVALUATION_FUNCTION, const_LP=50, POP_SIZE=150, MAX_GEN=2500, HOF=True, BOUNDS=[], STATISTICAL_LOG=True, GEN_SAVE=True, PARALLEL=False, SEED=None, SAVE_PATH = ""):
        self.LP = const_LP
        self.POP_SIZE = POP_SIZE
        self.MAX_GEN = MAX_GEN
        
        self.STATISTICAL_LOG = STATISTICAL_LOG
        self.GEN_SAVE = GEN_SAVE
        self.HOF = HOF

        self.SAVE_PATH = SAVE_PATH

        self.EVALUATION_FUNCTION = EVALUATION_FUNCTION
        
        self.config_seed = SEED if SEED is not None else int(1234*time.time())
        self.prng = Generator(MT19937(seed=self.config_seed))
        
        self.INITIAL_POP = []
        
        self.TOOLBOX = base.Toolbox()
        self.TOOLBOX.register("rand_1_bin", de_rand_1_bin)
        self.TOOLBOX.register("rand_to_best_2_bin", de_rand_to_best_2_bin)
        self.TOOLBOX.register("de_rand_2_bin", de_rand_2_bin)
        self.TOOLBOX.register("de_current_to_rand_1", de_current_to_rand_1)
        self.TOOLBOX.register("evaluate", evaluate) # aqui eu estou registrando a função que faz a avaliação do meu individuo
        self.TOOLBOX.register("rnd_selection", tools.selRandom)
        self.TOOLBOX.register("select_best", tools.selBest)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.CONFIGURED_CREATOR = creator
        
        if not BOUNDS:
            print("Error: BOUNDS argument cannot be initialized as an empty list! Initializing with random values!")
            BOUNDS = [(1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1), (1e-5, 9.5e-1)]

        self.l_bounds = [i[0] for i in BOUNDS]
        self.u_bounds = [i[1] for i in BOUNDS]
        
        self.print_config()
        self.gen_initial_pop()
        # self.print_gen(self.INITIAL_POP)

    def print_config(self):
        print("\n\n--------------------------------------------------------------------------------------------------------------------------")
        
        print("\t\t\t\t\tSaDE implementation by StormBreaker1726")
        print("Configuration:")
        print(f"\t Learning period = {self.LP}")
        print(f"\t Population size = {self.POP_SIZE}")
        print(f"\t Number of generations = {self.MAX_GEN}")
        print(f"\t Lower bounds = {self.l_bounds}")
        print(f"\t Upper bounds = {self.u_bounds}")
        print(f"\t Seed = {self.config_seed}")
        
        print("--------------------------------------------------------------------------------------------------------------------------\n\n")

    def gen_initial_pop(self):
        # Configuração do gerador da população inicial
        sampler = qmc.Sobol(d=len(self.l_bounds), scramble=True, rng=self.prng)
        m = np.log2(self.POP_SIZE)
        m = int(np.ceil(m))
        sample = sampler.random_base2(m=m)
        qmc.scale(sample=sample, l_bounds=self.l_bounds, u_bounds=self.u_bounds) # Garante que as minhas amostras estejam dentro do intervalo necessário

        for smp in sample:
            self.INITIAL_POP.append(self.CONFIGURED_CREATOR.Individual(smp.tolist()))

        # Realizando a evaluation do fitness
        for pos, ind in enumerate(self.INITIAL_POP):
            ind.fitness.values = self.TOOLBOX.evaluate(ind)

    def run_SaDE(self):
        pass
    
    def print_gen(self, pop):
        print("\n\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
        for pos, ind in enumerate(pop):
            print(f"\t Ind {pos}: Fitness = {ind.fitness.values} - Gene = {ind}")

        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
    
    def print_hof(self, hof):
        pass
    
    def save_gen(self, pop):
        pass
    
    def save_results(self):
        pass
    
    def clean(self):
        pass