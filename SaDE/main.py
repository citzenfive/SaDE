from SaDE_DEAP import *
from evaluation import *

s = SaDE(EVALUATION_FUNCTION=evaluate, const_LP=50, POP_SIZE=200, MAX_GEN=2000, HOF=True, STATISTICAL_LOG=True, GEN_SAVE=True, SEED=1234)