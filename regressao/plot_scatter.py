import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(X: np.ndarray, y: np.ndarray):
    """
    Plota um gráfico de dispersão da variável independente vs. variável dependente.

    Args:
        X (np.ndarray): Matriz de features (N×1).
        y (np.ndarray): Vetor de valores alvo (N×1).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.7, edgecolors='k')
    plt.xlabel("Velocidade do vento")
    plt.ylabel("Potência gerada")
    plt.title("Dispersão de velocidade do vento vs. potência gerada")
    plt.grid(True)
    plt.show()
