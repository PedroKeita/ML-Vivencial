"""
Exemplo de execução do módulo de classificação EMG.
"""

from classificacao import choose_k, view_scatter_plot, random_validator, loading_date
import pandas as pd

if __name__ == "__main__":

    #Carregar os dados
    x, y = loading_date()

    #Visualizar os dados
    view_scatter_plot(x, y)

    #Definição de valores de k para validação K-Fold
    k_values = [1, 7, 11, 17, 23, 39, 101, 501, 1001]
    better_k, results = choose_k(x, y, k_values)
    print("\nResultados com a validação K-Fold:")
    for k, score in results.items():
        print(f"k={k}: {score:.4f}")
    print(f"\nMelhor k encontrado foi: {better_k}")

    #Realizar a validação aleatória
    statistics, better_conf, worse_conf = random_validator(x, y, better_k)
    print("\nEstatísticas validação aleatória de 500 rodadas:")
    print(pd.DataFrame(statistics, index=["estatisticas"]).T)

    acc_best, conf_best = better_conf
    acc_worst, conf_worst = worse_conf

    print("\nMelhor caso de matriz de confusão (acurácia {:.4f}):".format(acc_best))
    print((conf_best))

    print("\nPior caso de matriz de confusão (acurácia {:.4f}):".format(acc_worst))
    print((conf_worst))
