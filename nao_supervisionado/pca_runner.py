import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from typing import Dict
import os

def pca_runner(X: np.ndarray, save_dir: str = "plots/pca", variances: list = [0.90, 0.80, 0.75]) -> Dict[float, np.ndarray]:
    os.makedirs(save_dir, exist_ok=True)
    
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
        
        file_path = os.path.join(save_dir, f"pca_{int(var*100)}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"PCA {int(var*100)}% salvo em: {os.path.abspath(file_path)}")
    
    return results
