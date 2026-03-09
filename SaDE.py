import numpy as np
import pandas as pd
import statsmodels as stt
import statistics as stat
import time

from scipy.stats import qmc

from numpy.random import Generator, MT19937, SeedSequence

from deap import base, creator, gp, tools

from mutation_cossover import *

class SaDE:
    def __init__(self, EVALUATION_FUNCTION, const_LP=50, POP_SIZE=150, MAX_GEN=2500, HOF_SIZE=1, BONDS=[], STATISTICAL_LOG=True, GEN_SAVE=True, PARALLEL=False, SEED=None, SAVE_PATH="", PARALLEL_MAP_FUNCTION=None):
        pass
    
    def print_configs(self):
        pass
    
    def _update_strategy_probabilities(self):
        pass
    
    def _gen_initial_pop(self):
        pass
    
    def run_SaDE(self):
        pass
    
    def _print_hof(self):
        pass
    
    def _save_gen(self):
        pass
    
    def save_results(self):
        pass
    
    def clean(self):
        pass