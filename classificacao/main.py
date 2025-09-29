"""
Execução do módulo de classificação de sinais EMG.
Testa KNN (com K-Fold e validação aleatória) e Gaussian Naive Bayes.
"""

from classificacao import (
    loading_date,
    view_scatter_plot,
    choose_k,
    random_validator,
    bayes_validator
)
import pandas as pd

if __name__ == "__main__":

    # Carregar os dados
    X, y = loading_date()

    # Visualizar os dados
    view_scatter_plot(X, y)

    # KNN
    k_values = [1, 7, 11, 17, 23, 39, 101, 501, 1001]
    better_k, results = choose_k(X, y, k_values)

    print("\nResultados com a validação K-Fold (KNN):")
    for k, score in results.items():
        print(f"k={k}: {score:.4f}")
    print(f"\nMelhor k encontrado foi: {better_k}")

    statistics_knn, better_conf_knn, worse_conf_knn = random_validator(X, y, better_k)
    print("\nEstatísticas KNN (500 rodadas de validação aleatória):")
    print(pd.DataFrame(statistics_knn, index=["estatísticas"]).T)

    acc_best, conf_best = better_conf_knn
    acc_worst, conf_worst = worse_conf_knn

    print("\nMelhor caso de matriz de confusão (KNN, acurácia {:.4f}):".format(acc_best))
    print(conf_best)

    print("\nPior caso de matriz de confusão (KNN, acurácia {:.4f}):".format(acc_worst))
    print(conf_worst)

    # Gaussian Naive Bayes
    statistics_bayes, better_conf_bayes, worse_conf_bayes = bayes_validator(X, y)

    print("\nEstatísticas Gaussian Naive Bayes (500 rodadas):")
    print(pd.DataFrame(statistics_bayes, index=["estatísticas"]).T)

    acc_best, conf_best = better_conf_bayes
    acc_worst, conf_worst = worse_conf_bayes

    print("\nMelhor caso de matriz de confusão (Bayes, acurácia {:.4f}):".format(acc_best))
    print(conf_best)

    print("\nPior caso de matriz de confusão (Bayes, acurácia {:.4f}):".format(acc_worst))
    print(conf_worst)
