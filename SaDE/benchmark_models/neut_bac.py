import numpy as np
import pandas as pd
from numpy import linalg as lng
from matplotlib import pyplot as plt
from scipy.integrate import odeint

def sys_bac_neutro(Y, t, cp, lnb, gn, lbn, mn, cn_max):
    '''
        Essa função retorna a derivada do sistema de equações diferenciais que modela a interação entre bactérias e neutrófilos.
    '''
    B, N = Y

    dBdt = cp*B - lnb*B*N
    dNdt = gn*B*(cn_max - N) - lbn*N*B - mn*N

    return [dBdt, dNdt]

def model(params):
    '''
    params[0] -> cp
    params[1] -> λnb
    params[2] -> γn
    params[3] -> λbn
    params[4] -> μn
    params[5] -> cn_max
    '''

    neutrophil_data = np.loadtxt('SaDE/benchmark_models/neutrophil_array.txt')
    bacteria_data = np.loadtxt('SaDE/benchmark_models/bacteria_array.txt')
    t = np.linspace(0, len(bacteria_data), len(bacteria_data))

    # print((params))
    
    cp     = params[0]
    lnb    = params[1]
    gn     = params[2]
    lbn    = params[3]
    mn     = params[4]
    cn_max = params[5]

    P0 = [bacteria_data[0], neutrophil_data[0]]
    mmargs = (cp, lnb, gn, lbn, mn, cn_max)

    try:
        # Temporarily treat RuntimeWarnings as errors to be caught
        with np.errstate(over='raise'):
            Y = odeint(sys_bac_neutro, P0 , t, args=(mmargs))

        # Check if the solver returned any NaN values, which also indicates failure
        if np.isnan(Y).any():
            raise FloatingPointError("Solver returned NaN values.")

        erro_B = lng.norm(Y[:,0] - bacteria_data, np.inf)
        erro_N = lng.norm(Y[:,1] - neutrophil_data, np.inf)
        error = max(erro_B, erro_N)

    # Catch the specific floating point errors that indicate instability
    except (FloatingPointError, ValueError): 
        # When the solver fails, return an infinite error.
        # This tells the optimizer to discard this parameter set.
        error = np.inf
        Y = None # No valid solution to return

    return [error, Y]

def neut_bac_model(ind):
    result = model(ind)
    return result[0],