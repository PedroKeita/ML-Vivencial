"""
Módulo para redução de dimensionalidade usando t-SNE.
Reduz os dados a 2 dimensões e salva gráficos de dispersão.
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def tsne_runner(X: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """
    Executa t-SNE para redução de dimensionalidade e plota o resultado.

    Args:
        X (np.ndarray): Dados normalizados.
        perplexity (int): Perplexidade do t-SNE.
        random_state (int): Semente aleatória.

    Returns:
        np.ndarray: Dados transformados em 2D.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], s=12, alpha=0.7)
    plt.title(f"t-SNE (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

    return X_tsne
