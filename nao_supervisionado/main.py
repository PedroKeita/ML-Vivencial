"""
Main para executar toda a lógica da pasta nao_supervisionado.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nao_supervisionado import (
    loading_images,
    normalize,
    tsne_runner,
    pca_runner,
    umap_runner,
    cluster_runner
)

def main():
    print("Carregando imagens...")
    X, files, labels = loading_images("data/RecFac")

    print("Normalizando dados...")
    X_norm = normalize(X)

    print("Executando t-SNE...")
    X_tsne = tsne_runner(X_norm)

    print("Executando PCA...")
    pca_results = pca_runner(X_norm)

    print("Executando UMAP...")
    umap_results = umap_runner(X_norm)

    print("Clusterizando com K-means no t-SNE...")
    kmeans_results = cluster_runner(X_tsne, k_values=[3, 5, 7], method="kmeans")

    print("Clusterizando com K-medoids no t-SNE...")
    kmedoids_results = cluster_runner(X_tsne, k_values=[3, 5, 7], method="kmedoids")

    print("\n Resultados K-means (Dunn):", kmeans_results)
    print("\nResultados K-medoids (Dunn):", kmedoids_results)

if __name__ == "__main__":
    main()
