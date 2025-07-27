import numpy as np
import pandas as pd
import ast
import subprocess

from numpy import linalg as lng

in_dir = "bcim_experimental/"

def run_bcim(ind):
    """
    Avalia um indivíduo executando um programa C externo.

    Esta versão espera que o programa C imprima DUAS linhas de números
    separados por espaços, que serão atribuídas a I_calc e E_calc.
    """
    
    caminho_executavel = "./serial_exec"
    try:
        comando = [caminho_executavel] + [str(gene) for gene in ind]
    except Exception as e:
        print(f"ERRO ao construir o comando. Indivíduo: {ind}. Erro: {e}")
        return 1e30, 1e30

    try:
        resultado_processo = subprocess.run(
            comando, capture_output=True, text=True, check=True, timeout=60
        )
        
        output_completo = resultado_processo.stdout

        # 1. Divide a string de saída em uma lista de linhas
        linhas = output_completo.strip().splitlines()
        
        # 2. Checagem de segurança: verifica se temos as duas linhas esperadas
        if len(linhas) < 2:
            print(f"ERRO: A saída do C não continha as 2 linhas esperadas.")
            print(f"   Saída recebida: '{output_completo}'")
            return 1e30, 1e30 # Penalidade

        # 3. Processa a primeira linha para I_calc
        try:
            # Pega a primeira linha, divide pelos espaços e converte cada parte para float
            I_calc = [float(s) for s in linhas[0].split()]
        except ValueError:
            print(f"ERRO: Não foi possível converter a primeira linha para uma lista de números.")
            print(f"   Linha 1 recebida: '{linhas[0]}'")
            return 1e30, 1e30

        # 4. Processa a segunda linha para E_calc
        try:
            E_calc = [float(s) for s in linhas[1].split()]
        except ValueError:
            print(f"ERRO: Não foi possível converter a segunda linha para uma lista de números.")
            print(f"   Linha 2 recebida: '{linhas[1]}'")
            return 1e30, 1e30
        return [I_calc, E_calc]

    except Exception as e:
        print(f"Ocorreu um erro geral durante a avaliação: {e}")
        return 1e30, 1e30

def evaluate(ind):
    DATA_IFN_GAMMA = pd.read_csv(
        in_dir+"ifn_gamma_IL-10_KO.csv", header=None
    )
    DATA_TNF_ALPHA = pd.read_csv(
        in_dir+"tnf_alpha_IL-10_KO.csv", header=None
    )
    DATA_IL_17 = pd.read_csv(in_dir+"il_17_IL-10_KO.csv", header=None)

    E_exp = 0.05882 * (DATA_TNF_ALPHA[1]).to_numpy()
    I_exp = 0.05882 * (DATA_IFN_GAMMA[1] + DATA_IL_17[1]).to_numpy()

    fitness_final = None

    error_e = None
    error_i = None
    
    I_calc, E_calc = run_bcim(ind)
    
    # error_e = lng.norm((E_exp-E_calc), np.inf)
    # error_i = lng.norm((I_exp-I_calc), np.inf)
    
    error_e = lng.norm((E_exp-E_calc), 2)
    error_i = lng.norm((I_exp-I_calc), 2)
    
    fitness_final = max(error_e, error_i)
    
    return fitness_final,