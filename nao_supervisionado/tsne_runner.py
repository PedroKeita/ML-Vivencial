import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def tsne_runner(X: np.ndarray, save_dir: str = "plots/tsne", perplexity: int = 30, random_state: int = 42) -> np.ndarray:
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
