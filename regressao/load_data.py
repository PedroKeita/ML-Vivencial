"""
Módulo para carregar dados.
"""
import pandas as pd
import os

def load_data(filepath: str):
    """
    Carrega o dataset de aerogerador.

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo de dados.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas "velocidade" e "potencia".
    """
    full_path = os.path.abspath(filepath)
    data = pd.read_csv(full_path, sep=r"\s+", header=None, names=["velocidade", "potencia"])
    return data
