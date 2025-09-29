"""
Módulo para clusterização de dados.
Suporta K-means e K-medoids, calcula índice de Dunn e salva gráficos 2D.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids 
from typing import Dict
from .dunn_index import dunn_index

def cluster_runner(X: np.ndarray, k_values: list = [3, 5, 7], method: str = "kmeans", plot: bool = True) -> Dict[int, float]:
    """
    Executa K-means ou K-medoids para diferentes valores de K e calcula o índice de Dunn.

    Args:
        X (np.ndarray): Dados a serem clusterizados.
        k_values (list, optional): Lista de valores de K a testar. Default é [3, 5, 7].
        method (str, optional): Método de clusterização, 'kmeans' ou 'kmedoids'. Default é 'kmeans'.
        plot (bool, optional): Se True, mostra o gráfico no notebook. Default é True.

    Returns:
        Dict[int, float]: Dicionário com cada K testado e seu respectivo índice de Dunn.
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
            if method == "kmeans":
                plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                            c="red", marker="x", s=100, label="Centroides")
            plt.title(f"{method.upper()} com K={k} (Dunn={dunn:.4f})")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            plt.legend()
            plt.show()
    
    return results
