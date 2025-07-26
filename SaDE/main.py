import multiprocessing
import sys

from SaDE_DEAP import *
from bcim import *
from benchmark import *



if __name__ == "__main__":
    N_PROCESSOR = int(sys.argv[1])
    
    pool = multiprocessing.Pool(processes=N_PROCESSOR)
    
    s = SaDE(
        EVALUATION_FUNCTION=neut_bac_model,
        const_LP=60,
        POP_SIZE=512,
        MAX_GEN=5000,
        STATISTICAL_LOG=True,
        HOF_SIZE=200,
        GEN_SAVE=True,
        # SEED=1234,
        BOUNDS=[(1e-5, 1e0), (1e-5, 1e0), (1e-5, 1e0), (1e-5, 1e0), (1e-5, 1e0), (1e-5, 1e0)],
        PARALLEL_MAP_FUNC=pool.map,
        ITERATIVE_SAVE=True,
        SAVE_PATH="adjustment/cluster_try/"
    )
    pool.close()
