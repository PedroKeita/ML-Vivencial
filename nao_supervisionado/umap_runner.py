"""
Módulo para redução de dimensionalidade usando UMAP.
Permite projeções em 2D e 3D para diferentes dimensões finais.
"""

import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def umap_runner(X: np.ndarray, dims: list = [3, 15, 55, 101], random_state: int = 42) -> Dict[int, np.ndarray]:
    """
    Executa UMAP para diferentes dimensões e plota a projeção 2D e 3D quando possível.

    Args:
        X (np.ndarray): Dados normalizados.
        dims (list): Lista de dimensões alvo.
        random_state (int): Semente aleatória.

    Returns:
        Dict[int, np.ndarray]: Dicionário com chave = dimensão e valor = dados transformados.
    """
    results = {}

    for d in dims:
        reducer = umap.UMAP(n_components=d, random_state=random_state)
        X_umap = reducer.fit_transform(X)
        results[d] = X_umap

        # Plot 2D
        plt.figure(figsize=(8,6))
        plt.scatter(X_umap[:,0], X_umap[:,1], s=12, alpha=0.7)
        plt.title(f"UMAP {d}D - projeção 2D")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.show()

        # Plot 3D 
        if d >= 3:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_umap[:,0], X_umap[:,1], X_umap[:,2], s=12, alpha=0.7)
            ax.set_title(f"UMAP {d}D - projeção 3D")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")
            plt.show()

    return results

