import umap.umap_ as umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from typing import Dict

def umap_runner(X: np.ndarray, dims: list = [3, 15, 55, 101], random_state: int = 42, 
                save_dir: str = "results/umap") -> Dict[int, np.ndarray]:

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for d in dims:
        reducer = umap.UMAP(n_components=d, random_state=random_state)
        X_umap = reducer.fit_transform(X)
        results[d] = X_umap

        # Sempre gera projeção 2D 
        plt.figure(figsize=(8,6))
        plt.scatter(X_umap[:,0], X_umap[:,1], s=12, alpha=0.7)
        plt.title(f"UMAP {d}D - projeção 2D")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"umap_{d}D_2d.png"))
        plt.close()

        # Sempre gera projeção 3D 
        if d >= 3:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_umap[:,0], X_umap[:,1], X_umap[:,2], s=12, alpha=0.7)
            ax.set_title(f"UMAP {d}D - projeção 3D")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"umap_{d}D_3d.png"))
            plt.close()

    return results
