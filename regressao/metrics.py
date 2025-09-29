"""
Módulo para cálculo de métricas estatísticas.
"""
import numpy as np

def calculate_statistics(values):
    """
    Calcula estatísticas básicas: média, desvio padrão, máximo e mínimo.

    Args:
        values (list or np.ndarray): Lista de valores.

    Returns:
        dict: Dicionário com média, desvio padrão, maior e menor valor.
    """
    
    return {
        "Média": np.mean(values),
        "Desvio-padrão": np.std(values, ddof=1),
        "Maior valor": np.max(values),
        "Menor valor": np.min(values)
    }
