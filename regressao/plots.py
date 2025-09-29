"""
Módulo para geração de gráficos.
"""
import matplotlib.pyplot as plt

def plot_scatter(x, y, xlabel="Velocidade do vento", ylabel="Potência gerada", title="Dispersão"):
    """
    Gera gráfico de dispersão.

    Parâmetros
    ----------
    x : array-like
        Valores do eixo X.
    y : array-like
        Valores do eixo Y.
    xlabel : str
        Label do eixo X.
    ylabel : str
        Label do eixo Y.
    title : str
        Título do gráfico.
    """
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
