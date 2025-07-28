import numpy as np
import pandas as pd
import statsmodels as stt
import time
import statistics as stat
import math
import os

from numpy.random import Generator, MT19937, SeedSequence

from scipy.stats import qmc

from deap import base
from deap import creator
from deap import gp
from deap import tools

from bcim import *
from mut_cross import *

class SaDE:
    def __init__(self, EVALUATION_FUNCTION, const_LP=50, POP_SIZE=150, MAX_GEN=2500, HOF_SIZE=1, BOUNDS=[], STATISTICAL_LOG=True, GEN_SAVE=True, PARALLEL=False, SEED=None, SAVE_PATH = "", PARALLEL_MAP_FUNC=None, ITERATIVE_SAVE=False):
        self.LP = const_LP
        self.POP_SIZE = POP_SIZE
        self.MAX_GEN = MAX_GEN
        
        self.STATISTICAL_LOG = STATISTICAL_LOG
        self.GEN_SAVE = GEN_SAVE
        self.SAVE_PATH = SAVE_PATH
        self.EVALUATION_FUNCTION = EVALUATION_FUNCTION
        
        self.ITERATIVE_SAVE = ITERATIVE_SAVE
        
        self.config_seed = SEED if SEED is not None else int(1234*time.time())
        self.prng = Generator(MT19937(seed=self.config_seed))
        
        self.INITIAL_POP = [] 
        
        random.seed(self.config_seed)

        # self.strategy_pool = [de_rand_1_bin, de_rand_to_best_2_bin, de_rand_2_bin, de_current_to_rand_1]
        self.strategy_pool = [de_rand_1_bin, de_rand_to_best_2_bin, de_rand_2_bin, de_current_to_rand_1, de_current_to_pbest_1, de_best_1_bin]
        self.num_strategies = len(self.strategy_pool)
        self.str_prob = np.full(self.num_strategies, 1.0 / self.num_strategies)
        self.success_counter = np.zeros(self.num_strategies)
        self.failure_counter = np.zeros(self.num_strategies)
        self.cr_memory = []
        self.crm = 0.5
        
        self.BEST = None
        
        self.TOOLBOX = base.Toolbox()
        
        if PARALLEL_MAP_FUNC:
            self.TOOLBOX.register("map", PARALLEL_MAP_FUNC)

        self.TOOLBOX.register("de_rand_1_bin", de_rand_1_bin)
        self.TOOLBOX.register("de_rand_to_best_2_bin", de_rand_to_best_2_bin)
        self.TOOLBOX.register("de_rand_2_bin", de_rand_2_bin)
        self.TOOLBOX.register("de_current_to_rand_1", de_current_to_rand_1)
        self.TOOLBOX.register("de_current_to_pbest_1", de_current_to_pbest_1)
        self.TOOLBOX.register("de_best_1_bin", de_best_1_bin)
    
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
        
        self.SAVE_PATH_GEN = self.SAVE_PATH+"gens/"
        
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        if not os.path.isdir(self.SAVE_PATH_GEN):
            os.mkdir(self.SAVE_PATH_GEN)
        
        self.TOP_TIER = int(0.05*self.POP_SIZE)
        
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
        """Gera a população inicial usando o amostrador Sobol e escalonamento manual."""
        
        # --- Passo 1: Configurar o amostrador ---
        sampler = qmc.Sobol(d=len(self.l_bounds), scramble=True, rng=self.prng)
        
        # --- Passo 2: Gerar a amostra no cubo unitário [0, 1] ---
        m = np.log2(self.POP_SIZE)
        m = int(np.ceil(m))
        sample_unitario = sampler.random_base2(m=m)
        # --- Passo 3: Escalonamento Manual (Substituindo qmc.scale) ---
        
        # Converte seus limites para arrays NumPy para permitir operações vetorizadas
        l_bounds_arr = np.array(self.l_bounds)
        u_bounds_arr = np.array(self.u_bounds)
        
        # Aplica a fórmula de escalonamento linear manualmente a toda a matriz de uma vez
        scaled_sample = l_bounds_arr + sample_unitario * (u_bounds_arr - l_bounds_arr)
        
        # --- Passo 4: Criar e avaliar os indivíduos do DEAP ---
        
        # Zera a população inicial antes de preenchê-la
        self.INITIAL_POP = []
        # Usa a nova variável 'scaled_sample'
        for smp in scaled_sample:
            self.INITIAL_POP.append(self.CONFIGURED_CREATOR.Individual(smp.tolist()))

        dbg_initial_pop = open(self.SAVE_PATH+"debug_pop_inicial.txt", "w+")
        # ## ADICIONE O PRINT DE VERIFICAÇÃO FINAL AQUI PARA CONFIRMAR ##
        print("\n--- DEBUG: VERIFICAÇÃO FINAL DA POPULAÇÃO INICIAL ---")
        dbg_initial_pop.write("\n--- DEBUG: VERIFICAÇÃO FINAL DA POPULAÇÃO INICIAL ---\n")
        
    
        # IMPORTANTE: Verifique se esta lista tem EXATAMENTE a mesma ordem
        # dos parâmetros na sua lista 'BOUNDS' que você passa para a classe.
        # Esta ordem é baseada no seu último print de 'Lower bounds'.
        nomes_dos_parametros = [
            'rho_i',
            'rho_e',
            'chi',       # Assumindo que o 3º parâmetro é chi
            'tau_b',
            'c_apcr_b',
            'c_apcm_b',  # Assumindo a ordem dos 6 otimizados
            'c_b_Th',    # Assumindo a ordem dos 4 novos
            'c_apc_c',
            'APCr_max'
        ]
    
        # Itera sobre cada parâmetro (cada coluna da população)
        for i in range(len(self.l_bounds)):
            param_name = nomes_dos_parametros[i]
            lower_bound = self.l_bounds[i]
            upper_bound = self.u_bounds[i]
            
            # Extrai todos os valores para o parâmetro atual de todos os indivíduos
            param_values = [ind[i] for ind in self.INITIAL_POP]
            
            print(f"\nParâmetro '{param_name}' (Bounds: [{lower_bound:.2e}, {upper_bound:.2e}]):")
            dbg_initial_pop.write(f"\nParâmetro '{param_name}' (Bounds: [{lower_bound:.2e}, {upper_bound:.2e}]):\n")
            
            if param_values: # Verifica se a lista não está vazia
                print(f"  Mínimo: {min(param_values):.6f}")
                print(f"  Máximo: {max(param_values):.6f}")
                print(f"  Média:  {np.mean(param_values):.6f}")
                dbg_initial_pop.write(f"  Mínimo: {min(param_values):.6f}\n")
                dbg_initial_pop.write(f"  Máximo: {max(param_values):.6f}\n")
                dbg_initial_pop.write(f"  Média:  {np.mean(param_values):.6f}\n")
            else:
                print("  Nenhum valor encontrado.")
                dbg_initial_pop.write("  Nenhum valor encontrado.\n")
                

        print("------------------------------------------------------\n")
        dbg_initial_pop.write("\n------------------------------------------------------\n")
        
        
        # exit(-1)
        # Avalia a população inicial (em paralelo, se configurado)
        fitnesses = self.TOOLBOX.map(self.TOOLBOX.evaluate, self.INITIAL_POP)
        for ind, fit in zip(self.INITIAL_POP, fitnesses):
            ind.fitness.values = fit
            
    
    def gen_new_rnd_pop(self, size):
        # Configuração do gerador da população inicial
        sampler = qmc.Sobol(d=len(self.l_bounds), scramble=True, rng=self.prng)
        m = np.log2(size)
        m = int(np.ceil(m))
        sample = sampler.random_base2(m=m)
        qmc.scale(sample=sample, l_bounds=self.l_bounds, u_bounds=self.u_bounds) # Garante que as minhas amostras estejam dentro do intervalo necessário

        new_rnd_pop = []

        for smp in sample:
            new_rnd_pop.append(self.CONFIGURED_CREATOR.Individual(smp.tolist()))

        # # Realizando a avaliação do fitness da população inicial
        # for pos, ind in enumerate(new_rnd_pop):
        #     ind.fitness.values = self.TOOLBOX.evaluate(ind)
        fitnesses = self.TOOLBOX.map(self.TOOLBOX.evaluate, new_rnd_pop)
        for ind, fit in zip(new_rnd_pop, fitnesses):
            ind.fitness.values = fit
        
        return new_rnd_pop.copy()
    
    def bound_maker(self, offspring):
        # pass
        for pos, par in enumerate(offspring):
            if par < self.l_bounds[pos]:
                par = self.l_bounds[pos]
            elif par > self.u_bounds[pos]:
                par = self.u_bounds[pos]
            else:
                par = par
            offspring[pos] = par

    def run_SaDE(self):
        self.save_config()
        print("\n\n--- Iniciando a execução do SaDE ---")
        # Use os atributos inicializados no __init__
        STOP_CRITERIA = int(0.30 * self.MAX_GEN)
        NO_IMP_GEN = 0
        CURRENT_POPULATION = self.INITIAL_POP.copy()
        ESTAG_POP = 0
        
        #======================================================================
        # LAÇO PRINCIPAL DE GERAÇÕES
        #======================================================================
        for GEN in range(self.MAX_GEN):
            init_gen = time.time()
            # --- Preparação da Geração ---
            CURRENT_BEST = self.TOOLBOX.select_best(CURRENT_POPULATION, 1)[0]
            self.HOF.update(CURRENT_POPULATION)

            # --- FASE 1: Geração de Todos os Candidatos (sem avaliar) ---
            trial_vectors_info = []
            for IDV in CURRENT_POPULATION:
                # Gera parâmetros e sorteia a estratégia para este indivíduo
                current_F = self.prng.normal(0.5, 0.3)
                current_CR = np.clip(self.prng.normal(self.crm, 0.1), 0.0, 1.0)
                str_index = self.prng.choice(self.num_strategies, p=self.str_prob)
                
                chosen_strategy_func = self.strategy_pool[str_index]

                # Prepara os argumentos para a função de mutação
                args = {
                    'ind': IDV, 'population': CURRENT_POPULATION, 'f': current_F, 'cr': current_CR,
                    'creator': self.CONFIGURED_CREATOR, 'toolbox': self.TOOLBOX
                }
                # if chosen_strategy_func.__name__ == 'de_rand_to_best_2_bin':
                #     args['best'] = CURRENT_BEST
                if chosen_strategy_func.__name__ in ['de_rand_to_best_2_bin', 'de_best_1_bin']:
                    args['best'] = CURRENT_BEST
                
                # Gera o candidato e aplica os limites
                offspring_candidate = chosen_strategy_func(**args)
                # self.bound_maker(offspring=offspring_candidate)
                if any(x < 0 for x in offspring_candidate) or any(x < self.l_bounds[pos] for pos, x in enumerate(offspring_candidate)) or any(x > self.u_bounds[pos] for pos, x in enumerate(offspring_candidate)):
                    trial_vectors_info.append({
                        "vector": IDV,
                        "strategy_index": str_index,
                        "cr": current_CR
                    })
                else:
                    # Guarda o candidato e as informações usadas para gerá-lo.
                    # O fitness ainda não foi calculado.
                    trial_vectors_info.append({
                        "vector": offspring_candidate,
                        "strategy_index": str_index,
                        "cr": current_CR
                    })

            # --- FASE 2: Avaliação em Paralelo (O Lote) ---
            
            # Extrai apenas a lista de indivíduos que precisam ser avaliados
            candidates_to_eval = [info["vector"] for info in trial_vectors_info]
            
            other_operations_init = time.time()

            # AQUI ACONTECE A MÁGICA: O 'map' distribui o trabalho entre os processadores
            fitnesses = self.TOOLBOX.map(self.TOOLBOX.evaluate, candidates_to_eval)
            
            # Atribui os fitness calculados de volta aos seus respectivos indivíduos
            for ind, fit in zip(candidates_to_eval, fitnesses):
                ind.fitness.values = fit

            other_operations_end = time.time()
            # --- FASE 3: Seleção e Contagem ---
            other_operations_total = other_operations_end-other_operations_init
            
            next_gen = []
            for i in range(self.POP_SIZE):
                target_vector = CURRENT_POPULATION[i]
                trial_info = trial_vectors_info[i]
                trial_vector = trial_info["vector"] # Agora já tem o fitness calculado
                
                # Lógica de seleção um-para-um
                if trial_vector.fitness.values <= target_vector.fitness.values:
                    next_gen.append(trial_vector)
                    # Registra o sucesso para a estratégia e o CR usados
                    self.success_counter[trial_info["strategy_index"]] += 1
                    self.cr_memory.append(trial_info["cr"])
                else:
                    next_gen.append(target_vector)
                    # Registra a falha para a estratégia usada
                    self.failure_counter[trial_info["strategy_index"]] += 1
            
            # Atualiza a população para a próxima geração
            CURRENT_POPULATION = next_gen.copy()
            # --- FASE 4: Lógica de Adaptação do SaDE ---
            
            if (GEN + 1) % self.LP == 0:
                self._update_strategy_probabilities()
                
                self.success_counter.fill(0)
                self.failure_counter.fill(0)
                
                if self.cr_memory:
                    self.crm = stat.median(self.cr_memory)
                self.cr_memory = []

            end_gen = time.time()
            
            if GEN % 10 == 0:
                print(f"Gen {GEN}: Best Fitness = {CURRENT_BEST.fitness.values[0]:.6f} | "
                    f"Probabilities: {[f'{p:.2f}' for p in self.str_prob]} | Spent time = {end_gen-init_gen} seconds | Eval operations = {other_operations_total} seconds | No improvement generations = {NO_IMP_GEN} | std = {self.stats.compile(CURRENT_POPULATION)['std']} | Estagnation = {ESTAG_POP}")
                if self.ITERATIVE_SAVE == True:
                    self.save_gen(CURRENT_POPULATION, GEN, end_gen-init_gen)

            PROB_NEW_BEST = tools.selBest(next_gen, 1)[0]
            if math.isclose(PROB_NEW_BEST.fitness.values[0], CURRENT_BEST.fitness.values[0], rel_tol=1e-9, abs_tol=1e-6):
                NO_IMP_GEN += 1
            else:
                NO_IMP_GEN = 0
            if NO_IMP_GEN > STOP_CRITERIA:
                print("Parando por estagnação!")
                self.save_gen(CURRENT_POPULATION, GEN, end_gen-init_gen)
                self.save_gen(next_gen, GEN+1, 0)
                break
            if math.isclose(PROB_NEW_BEST.fitness.values[0], 0.0, rel_tol=1e-7, abs_tol=1e-7):
                print("Parando por melhora absoluta!")
                self.save_gen(CURRENT_POPULATION, GEN, end_gen-init_gen)
                self.save_gen(next_gen, GEN+1, 0)
                break
            if math.isclose(self.stats.compile(CURRENT_POPULATION)['std'], 1e-6, rel_tol=1e-6, abs_tol=1e-6) and ESTAG_POP >= 3:
                print("Parando por uniformização da população!")
                self.save_gen(CURRENT_POPULATION, GEN, end_gen-init_gen)
                self.save_gen(next_gen, GEN+1, 0)
                break
            elif math.isclose(self.stats.compile(CURRENT_POPULATION)['std'], 1e-6, rel_tol=1e-6, abs_tol=1e-6) and ESTAG_POP < 3:
                ESTAG_POP += 1
                print(f"Reinicializando população em gen = {GEN}!")
                TOP_TIER_POP = []
                TOP_TIER_POP = tools.selBest(next_gen, self.TOP_TIER)
                new_rnd_pop = self.gen_new_rnd_pop(self.POP_SIZE - self.TOP_TIER)
                new_rnd_pop = TOP_TIER_POP + new_rnd_pop
                CURRENT_POPULATION = new_rnd_pop.copy()
                self.str_prob = np.full(self.num_strategies, 1.0 / self.num_strategies)
                print(f"Gen {GEN}: Best Fitness = {CURRENT_BEST.fitness.values[0]:.6f} | "
                    f"Probabilities: {[f'{p:.2f}' for p in self.str_prob]} | Spent time = {end_gen-init_gen} seconds | Eval operations = {other_operations_total} seconds | No improvement generations = {NO_IMP_GEN} | std = {self.stats.compile(CURRENT_POPULATION)['std']} | Estagnation = {ESTAG_POP}")
        # --- FIM DO LAÇO PRINCIPAL ---

        # Guarda o melhor indivíduo final
        self.BEST = self.TOOLBOX.select_best(CURRENT_POPULATION, 1)[0]



    def print_gen(self, pop):
        print("\n\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
        for pos, ind in enumerate(pop):
            print(f"\t Ind {pos}: Fitness = {ind.fitness.values} - Gene = {ind}")

        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
    
    def print_hof(self, hof):
        pass
    
    def save_gen(self, pop, gen_num, time):
        BEST_GEN = self.TOOLBOX.select_best(pop, 1)[0]
        gen_file = open(self.SAVE_PATH_GEN+f"gen_{gen_num}.dat","w+")
        gen_file.write(f"\tBest individual: \n\t\tGene: {BEST_GEN}\n\t\tFitness: {BEST_GEN.fitness.values[0]} \n\n\nGeneration: {gen_num}\n\n\nSpent time: {time} seconds\n\n")
        for pos, ind in enumerate(pop):
            gen_file.write(f"\tIndividual {pos}: \n\t\tGene: {ind}\n\t\tFitness: {ind.fitness.values[0]} \n\n")
        gen_file.close()

    def save_config(self):
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
