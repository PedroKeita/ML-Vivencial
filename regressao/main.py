"""
Main para executar a regressão linear do aerogerador.
"""
import os
import numpy as np
import pandas as pd

from regressao.load_data import load_data
from regressao.model_runner import run_linear_regression
from regressao.metrics import calculate_statistics

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Carregar dados
    data = load_data("../data/aerogerador.dat")
    x = data[["velocidade"]].values
    y = data["potencia"].values

    # Executar regressão linear
    results = run_linear_regression(x, y, n_rounds=500)

    # Estatísticas
    tabela = {metric: calculate_statistics(values) for metric, values in results.items()}

    print("\nResultados Regressão Linear considerando 500 rodadas:")
    print(pd.DataFrame(tabela).T)

if __name__ == "__main__":
    main()
