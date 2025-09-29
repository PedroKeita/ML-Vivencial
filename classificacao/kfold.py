"""
Função para escolher o melhor k para K-NN usando validação K-Fold.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

def choose_k(X, y, k_values):
    """
    Encontra o melhor valor de K para K-NN via K-Fold Cross Validation.

    Args:
        X (np.ndarray): Matriz de características N×p
        y (np.ndarray): Vetor de classes N×1
        k_values (list): Lista de valores de k a testar

    Returns:
        better_k (int): Melhor valor de K
        results (dict): Dicionário com média das acurácias para cada k
    """

    results = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=kf)
        results[k] = scores.mean()
    
    better_k = max(results, key=results.get)
    return better_k, results