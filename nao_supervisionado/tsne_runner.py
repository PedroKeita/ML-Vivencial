"""
Módulo para redução de dimensionalidade usando t-SNE.
Reduz os dados a 2 dimensões e salva gráficos de dispersão.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def tsne_runner(X: np.ndarray, save_dir: str = "plots/tsne", perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """
    Aplica t-SNE para reduzir os dados a 2 dimensões e salva o gráfico.

    Args:
        X (np.ndarray): Dados normalizados.
        save_dir (str): Diretório onde salvar os plots.
        perplexity (int): Valor de perplexity do t-SNE.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        np.ndarray: Dados reduzidos em 2 dimensões.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], s=12, alpha=0.7)
    plt.title(f"t-SNE (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    file_path = os.path.join(save_dir, f"tsne_perplexity_{perplexity}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"t-SNE salvo em: {os.path.abspath(file_path)}")
    
    return X_tsne
