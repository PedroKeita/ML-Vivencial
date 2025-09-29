"""
Visualização de dispersão dos sinais EMG.
"""

import matplotlib.pyplot as plt

def view_scatter_plot(x,y):
    """
    Plota gráfico de dispersão das amostras de EMG destacando as classes.

    Args:
        x (np.ndarray): Matriz de características N×p
        y (np.ndarray): Vetor de classes N×1
    """
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.7)
    plt.xlabel("Sensor 1: Corrugador do Supercílio")
    plt.ylabel("Sensor 2: Zigomático Maior")
    plt.title("Dispersão dos sinais EMG por 2 sensores")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()