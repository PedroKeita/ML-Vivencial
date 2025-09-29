"""
Módulo para clusterização de dados.
Suporta K-means e K-medoids, calcula índice de Dunn e salva gráficos 2D.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids 
from typing import Dict
from .dunn_index import dunn_index

def cluster_runner(X: np.ndarray, k_values: list = [3, 5, 7], method: str = "kmeans", plot: bool = True) -> Dict[int, float]:
    """
    Executa K-means ou K-medoids para múltiplos valores de K e calcula índice de Dunn.

    Args:
        X (np.ndarray): Dados de entrada (N amostras x p características).
        k_values (list): Lista de valores K para testar.
        method (str): "kmeans" ou "kmedoids".
        plot (bool): Se True, gera e salva gráficos 2D (somente se X tiver 2 dimensões).

    Returns:
        Dict[int,float]: Dicionário mapeando cada K para seu índice de Dunn.
    """
    
    results = {}
    for k in k_values:
        if method == "kmeans":
            model = KMeans(n_clusters=k, random_state=42)
        elif method == "kmedoids":
            model = KMedoids(n_clusters=k, random_state=42)
        else:
            raise ValueError("Método deve ser 'kmeans' ou 'kmedoids'.")

        labels = model.fit_predict(X)
        dunn = dunn_index(X, labels)
        results[k] = dunn

        if plot and X.shape[1] == 2:
            plt.figure(figsize=(7, 5))
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7, s=20)
            plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", marker="x", s=100)
            plt.title(f"{method.upper()} com K={k} (Dunn={dunn:.4f})")
            plt.savefig(f"plots/{method}_k{k}.png")
            plt.close()

    return results
