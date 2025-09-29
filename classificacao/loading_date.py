"""
Função para carregar o dataset EMGDataset.csv.
O dataset possui 50000 amostras de sinais EMG de dois músculos faciais
e uma coluna de classe correspondente à expressão facial.
"""

import numpy as np
import pandas as pd
import os

def loading_date(path="data/EMGDataset.csv"):

    """
    Carrega os dados de sinais EMG do arquivo CSV.

    Args:
        path (str): Caminho relativo do arquivo CSV.

    Returns:
        X (np.ndarray): Matriz de características N×p (N amostras, p sensores)
        y (np.ndarray): Vetor de classes N×1
    """

    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(base_dir)               
    file_path = os.path.join(project_root, path)  

    data = pd.read_csv(file_path, header=None)

    # Sensor 1 e Sensor 2
    X = data.iloc[:, :2].values   

    # Classes
    y = data.iloc[:, 2].values.astype(int)    

    return X,y
