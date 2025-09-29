"""
Módulo para redução de dimensionalidade usando PCA.
Mantém diferentes proporções de variância e salva gráficos 2D.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from typing import Dict

def pca_runner(X: np.ndarray, variances: list = [0.90, 0.80, 0.75]) -> Dict[float, np.ndarray]:
    """
    Executa PCA para diferentes percentuais de variância e plota os resultados.

    Args:
        X (np.ndarray): Dados normalizados.
        variances (list): Percentuais de variância a preservar.

    Returns:
        Dict[float, np.ndarray]: Dicionário com chave = variância e valor = dados transformados.
    """
    results = {}
    for var in variances:
        pca = PCA(var, random_state=42)
        X_pca = pca.fit_transform(X)
        results[var] = X_pca

        plt.figure(figsize=(7,5))
        plt.scatter(X_pca[:,0], X_pca[:,1], s=12, alpha=0.7)
        plt.title(f"PCA - {int(var*100)}% da variância ({X_pca.shape[1]} componentes)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

    return results
