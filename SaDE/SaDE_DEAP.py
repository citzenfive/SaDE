import numpy as np
import pandas as pd
import statsmodels as stt
import time
import statistics as stat

from numpy.random import Generator, MT19937, SeedSequence

from scipy.stats import qmc

from deap import base
from deap import creator
from deap import gp
from deap import tools

from evaluation import *
from mut_cross import *

class SaDE:
    def __init__(self, EVALUATION_FUNCTION, const_LP=50, POP_SIZE=150, MAX_GEN=2500, HOF_SIZE=1, BOUNDS=[], STATISTICAL_LOG=True, GEN_SAVE=True, PARALLEL=False, SEED=None, SAVE_PATH = ""):
        self.LP = const_LP
        self.POP_SIZE = POP_SIZE
        self.MAX_GEN = MAX_GEN
        
        self.STATISTICAL_LOG = STATISTICAL_LOG
        self.GEN_SAVE = GEN_SAVE
        self.SAVE_PATH = SAVE_PATH
        self.EVALUATION_FUNCTION = EVALUATION_FUNCTION
        
        self.config_seed = SEED if SEED is not None else int(1234*time.time())
        self.prng = Generator(MT19937(seed=self.config_seed))
        
        self.INITIAL_POP = [] 
        

        self.strategy_pool = [de_rand_1_bin, de_rand_to_best_2_bin, de_rand_2_bin, de_current_to_rand_1]
        self.num_strategies = len(self.strategy_pool)
        self.str_prob = np.full(self.num_strategies, 1.0 / self.num_strategies)
        self.success_counter = np.zeros(self.num_strategies)
        self.failure_counter = np.zeros(self.num_strategies)
        self.cr_memory = []
        self.crm = 0.5
        
        self.BEST = None
        
        self.TOOLBOX = base.Toolbox()

        self.TOOLBOX.register("rand_1_bin", self.strategy_pool[0])
        self.TOOLBOX.register("rand_to_best_2_bin", self.strategy_pool[1])
        self.TOOLBOX.register("rand_2_bin", self.strategy_pool[2])
        self.TOOLBOX.register("current_to_rand_1", self.strategy_pool[3])
        self.TOOLBOX.register("evaluate", self.EVALUATION_FUNCTION)
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
        
        self.HOF = tools.HallOfFame(HOF_SIZE)
        
        self.print_config()
        self.INIT_TIME = time.time()
        self.gen_initial_pop()
        self.HOF.update(self.INITIAL_POP)
        self.run_SaDE()
        self.END_TIME = time.time()
        self.save_results()
        
        print("\n\n--------------------------------------------------------------------------------------------------------------------------")
        print(f"\n\nBest individual {self.BEST} with fitness {self.BEST.fitness.values[0]}\n\n")
        print("--------------------------------------------------------------------------------------------------------------------------\n\n")

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

    def _update_strategy_probabilities(self):
        """
        Calcula e normaliza as probabilidades das estratégias de forma segura.

        Esta versão garante que:
        1. Não ocorra divisão por zero se uma estratégia nunca for usada.
        2. As probabilidades sejam resetadas se nenhuma estratégia tiver sucesso.
        3. Nenhuma estratégia tenha sua probabilidade zerada, garantindo uma
        probabilidade mínima de sobrevivência (p_min).
        """
        # --- Passo 1: Calcular a taxa de sucesso de forma segura ---

        # Garante que os contadores são arrays NumPy para cálculos vetorizados
        success = np.array(self.success_counter, dtype=float)
        failure = np.array(self.failure_counter, dtype=float)
        
        total_attempts = success + failure
        
        # Inicializa as taxas de sucesso como um vetor de zeros
        success_rate = np.zeros(self.num_strategies)
        
        # Calcula a taxa de sucesso apenas onde o total de tentativas for maior que zero
        np.divide(success, total_attempts, out=success_rate, where=total_attempts != 0)

        # --- Passo 2: Tratar o caso extremo (nenhum sucesso) ---

        total_rate_sum = np.sum(success_rate)
        
        # Se a soma das taxas for zero, nenhuma estratégia funcionou.
        # Reseta as probabilidades para um estado uniforme e termina a função.
        if total_rate_sum == 0:
            self.str_prob = np.full(self.num_strategies, 1.0 / self.num_strategies)
            return

        # --- Passo 3: Garantir a Probabilidade Mínima (p_min) ---

        # Define a probabilidade mínima de sobrevivência para cada estratégia
        p_min = 0.05 
        
        # Calcula as probabilidades brutas com base no sucesso relativo
        raw_probabilities = success_rate / total_rate_sum
        
        # Garante que nenhuma probabilidade seja menor que p_min.
        # np.maximum compara cada probabilidade com p_min e escolhe o maior valor.
        corrected_probabilities = np.maximum(raw_probabilities, p_min)
        
        # Re-normaliza os valores para garantir que a soma de todas as probabilidades seja 1.
        # Isso é necessário porque ao elevar os valores para p_min, a soma total > 1.
        final_probabilities = corrected_probabilities / np.sum(corrected_probabilities)
        
        # Atualiza o atributo de probabilidade da classe
        self.str_prob = final_probabilities

    def gen_initial_pop(self):
        # Configuração do gerador da população inicial
        sampler = qmc.Sobol(d=len(self.l_bounds), scramble=True, rng=self.prng)
        m = np.log2(self.POP_SIZE)
        m = int(np.ceil(m))
        sample = sampler.random_base2(m=m)
        qmc.scale(sample=sample, l_bounds=self.l_bounds, u_bounds=self.u_bounds) # Garante que as minhas amostras estejam dentro do intervalo necessário

        for smp in sample:
            self.INITIAL_POP.append(self.CONFIGURED_CREATOR.Individual(smp.tolist()))

        # Realizando a avaliação do fitness da população inicial
        for pos, ind in enumerate(self.INITIAL_POP):
            ind.fitness.values = self.TOOLBOX.evaluate(ind)
    
    def bound_maker(self, offspring):
        # pass
        for pos, par in enumerate(offspring):
            par = (self.l_bounds[pos]) + par * (self.u_bounds[pos] - self.l_bounds[pos])
            offspring[pos] = par

    def run_SaDE(self):
        # pass
        print("\n\n")
        ''' rand_1_bin, rand_to_best_2_bin, rand_2_bin, current_to_rand_1 '''
        STOP_CRITERIA = int(0.15 * self.MAX_GEN)
        NO_IMP_GEN = 0
        CURRENT_POPULATION = self.INITIAL_POP.copy()
        
        for GEN in range(self.MAX_GEN):
            CURRENT_BEST = tools.selBest(CURRENT_POPULATION, 1)[0]
            if(GEN % 100 == 0):
                print(CURRENT_BEST)
                print(CURRENT_BEST.fitness.values[0])
                print(self.str_prob)
                print("\n\n")
            next_gen = []
            self.HOF.update(CURRENT_POPULATION)
            
            # Periodo de Aprendizado
            for IDV in CURRENT_POPULATION:
                current_F = self.prng.normal(0.5, 0.3)
                current_CR = np.clip(self.prng.normal(self.crm, 0.1), 0.0, 1.0)
                
                strategy = self.prng.uniform(0, 1)
                
                str_index = self.prng.choice(self.num_strategies, p=self.str_prob)
                chosen_strategy_func = self.strategy_pool[str_index]

                args = {
                    'ind': IDV, 'population': CURRENT_POPULATION, 'f': current_F, 'cr': current_CR,
                    'creator': self.CONFIGURED_CREATOR, 'toolbox': self.TOOLBOX
                }
                if chosen_strategy_func.__name__ == 'de_rand_to_best_2_bin':
                    args['best'] = CURRENT_BEST
                
                offspring_candidate = chosen_strategy_func(**args)
                
                self.bound_maker(offspring=offspring_candidate)
                
                if len(offspring_candidate) == 0:
                    print(len(offspring_candidate))
                    print(offspring_candidate)
                    print("Offspring vazio")
                
                if any(n < 0 for n in offspring_candidate):
                    next_gen.append(IDV)
                    self.failure_counter[str_index] += 1
                else:
                    offspring_candidate.fitness.values = self.TOOLBOX.evaluate(offspring_candidate)
                    if offspring_candidate.fitness.values <= IDV.fitness.values:
                        next_gen.append(offspring_candidate)
                        self.success_counter[str_index] += 1
                        self.cr_memory.append(current_CR)
                    else:
                        next_gen.append(IDV)
                        self.failure_counter[str_index] += 1
            
            # Período de atualização
            if (GEN+1) % self.LP == 0:
                self._update_strategy_probabilities()
                
                self.success_counter = np.zeros(self.num_strategies)
                self.failure_counter = np.zeros(self.num_strategies)
                
                if self.cr_memory:
                    self.crm = stat.median(self.cr_memory)
                self.cr_memory = []

            CURRENT_POPULATION = next_gen.copy()

            PROB_NEW_BEST = tools.selBest(next_gen, 1)[0]
            if PROB_NEW_BEST.fitness.values[0] == (CURRENT_BEST.fitness.values[0] + 1.0e-6) or PROB_NEW_BEST.fitness.values[0] == (CURRENT_BEST.fitness.values[0] - 1.0e-6):
                NO_IMP_GEN += 1
            if NO_IMP_GEN > STOP_CRITERIA:
                print("Parando por estagnação!")
                break

        self.BEST = tools.selBest(CURRENT_POPULATION, 1)[0]



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
        best_file = open(self.SAVE_PATH+"BEST_OF_ALL.dat","w+")
        HOF_file = open(self.SAVE_PATH+"HOF.dat","w+")
        config_file = open(self.SAVE_PATH+"configurations.dat","w+")
        
        config_file.write("--------------------------------------------------------------------------------------------------------------------------\n")
        config_file.write("\t\t\t\t\tSaDE implementation by StormBreaker1726\n")
        config_file.write("Configuration:\n")
        config_file.write(f"\t Learning period = {self.LP}\n")
        config_file.write(f"\t Population size = {self.POP_SIZE}\n")
        config_file.write(f"\t Number of generations = {self.MAX_GEN}\n")
        config_file.write(f"\t Lower bounds = {self.l_bounds}\n")
        config_file.write(f"\t Upper bounds = {self.u_bounds}\n")
        config_file.write(f"\t Seed = {self.config_seed}\n")
        config_file.write("--------------------------------------------------------------------------------------------------------------------------\n\n")
        config_file.close()
        
        best_file.write("--------------------------------------------------------------------------------------------------------------------------\n")
        best_file.write(f"Optimization for {len(self.l_bounds)} parameters\n")
        best_file.write(f"Fitness value: {self.BEST.fitness.values[0]}\n")
        best_file.write(f"Parameters optimized: {self.BEST}\n")
        best_file.write(f"Spent time: {self.END_TIME - self.INIT_TIME} seconds\n")
        best_file.write("--------------------------------------------------------------------------------------------------------------------------\n")
        best_file.close()
        
        HOF_file.write("--------------------------------------------------------------------------------------------------------------------------\n")
        HOF_file.write(f"Hall of Fame with {len(self.HOF)} individuals\n")
        for pos, ind in enumerate(self.HOF):
            HOF_file.write(f"\tIndividual {pos}: \n\t\tGene: {ind}\n\t\tFitness: {ind.fitness.values[0]} \n")
        HOF_file.write("\n--------------------------------------------------------------------------------------------------------------------------\n")
        HOF_file.close()
    
    def clean(self):
        pass