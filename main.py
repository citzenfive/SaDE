import multiprocessing
import sys

from SaDE import *

from run_benchmark import *

def teste():
    pass

if __name__ == "__main__":
    N_PROCESSOR = int(sys.argv[1])
    SAVE_PTH = sys.argv[2]
    
    pool = multiprocessing.Pool(processes=N_PROCESSOR)
    
    BOUNDS=[
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
        (1e-5, 1e0),
    ]
    
    this_sade = SaDE(EVALUATION_FUNCTION=neut_bac_model, PARALLEL=True, NUM_PAR=6, SAVE_PATH=SAVE_PTH, PARALLEL_MAP_FUNCTION=pool.map, BOUNDS=BOUNDS)
    this_sade.print_config()
    this_sade.run_SaDE()
    this_sade.save_config()
    
    
    pool.close()