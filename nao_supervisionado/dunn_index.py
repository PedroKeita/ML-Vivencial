"""
Módulo para cálculo do índice de Dunn.
Permite avaliar a qualidade de clusters (quanto maior, melhor a separação).
"""

import numpy as np
from scipy.spatial.distance import cdist

def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula o índice de Dunn para avaliar a qualidade de clusters.

    Args:
        X (np.ndarray): Dados (N amostras x p características).
        labels (np.ndarray): Labels atribuídos pelo método de clusterização.

    Returns:
        float: Índice de Dunn (quanto maior, melhor a separação entre clusters).
    """
    
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return 0.0

    # Distância mínima inter-cluster
    min_intercluster = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            points_i = X[labels == clusters[i]]
            points_j = X[labels == clusters[j]]
            dist = np.min(cdist(points_i, points_j))
            min_intercluster = min(min_intercluster, dist)

    # Distância máxima intra-cluster
    max_intracluster = 0
    for c in clusters:
        points = X[labels == c]
        if len(points) > 1:
            dist = np.max(cdist(points, points))
            max_intracluster = max(max_intracluster, dist)

    return min_intercluster / (max_intracluster + 1e-6)
