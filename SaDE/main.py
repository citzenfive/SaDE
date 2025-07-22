from SaDE_DEAP import *
from benchmark import *
from evaluation import *

s = SaDE(
    EVALUATION_FUNCTION=neut_bac_model,
    const_LP=60,
    POP_SIZE=200,
    MAX_GEN=5000,
    STATISTICAL_LOG=True,
    HOF_SIZE=100,
    GEN_SAVE=True,
    BOUNDS=[
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
    ],
)
