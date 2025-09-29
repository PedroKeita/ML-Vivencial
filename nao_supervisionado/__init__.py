"""
Pacote nao_supervisionado

Módulo com funcionalidades para análise exploratória de dados de imagens faciais.
Inclui:
- Carregamento de imagens
- Normalização
- Redução de dimensionalidade (t-SNE, PCA, UMAP)
- Clusterização (K-means, K-medoids)
- Cálculo do índice de Dunn
"""

from .loading_images import loading_images
from .normalize import normalize
from .tsne_runner import tsne_runner
from .pca_runner import pca_runner
from .umap_runner import umap_runner
from .cluster_runner import cluster_runner
from .dunn_index import dunn_index

__all__ = [
    "loading_images",
    "normalize",
    "tsne_runner",
    "pca_runner",
    "umap_runner",
    "cluster_runner",
    "dunn_index"
]