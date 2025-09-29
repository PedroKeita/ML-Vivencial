"""
Módulo de classificação de sinais EMG faciais utilizando K-NN.
Fornece funcionalidades para:
- Carregar os dados
- Visualizar os dados
- Escolher o melhor valor de k via validação K-Fold
- Avaliar modelo com validação aleatória (random sampling)
"""

from .loading_date import loading_date
from .kfold import choose_k
from .random_validator import random_validator
from .view import view_scatter_plot

__all__ = [
    "loading_date",
    "choose_k",
    "random_validator",
    "view_scatter_plot",
]