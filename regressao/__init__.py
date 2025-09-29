"""
Pacote regressao

Módulo para execução de regressão linear no dataset de aerogerador.
Inclui carregamento de dados, treinamento, métricas e visualização.
"""

from .load_data import load_data
from .model_runner import run_linear_regression
from .metrics import calculate_statistics
from .plot_scatter import plot_scatter

