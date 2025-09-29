"""
Módulo para normalização de dados.
Aplica StandardScaler (média=0, std=1) aos dados de entrada.
"""

from sklearn.preprocessing import StandardScaler
import numpy as np

def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normaliza os dados utilizando StandardScaler.

    Args:
        X (np.ndarray): Array de entrada (N amostras x p características).

    Returns:
        np.ndarray: Dados normalizados.
    """

    scaler = StandardScaler()
    return scaler.fit_transform(X)