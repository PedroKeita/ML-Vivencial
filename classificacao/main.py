from classificacao import choose_k, view_scatter_plot, random_validator, loading_date
import pandas as pd

if __name__ == "__main__":
    x, y = loading_date()

    view_scatter_plot(x, y)

    k_values = [1, 7, 11, 17, 23, 39, 101, 501, 1001]
    better_k, results = choose_k(x, y, k_values)
    print("\nResultados com a validação K-Fold:")
    for k, score in results.items():
        print(f"k={k}: {score:.4f}")
    print(f"\nMelhor k encontrado foi: {better_k}")

    statistics, better_conf, worse_conf = random_validator(x, y, better_k)
    print("\nEstatísticas validação aleatória de 500 rodadas:")
    print(pd.DataFrame(statistics, index=[0]).T)

    print("\nMelhor caso de matriz de confusão:")
    print(better_conf)

    print("\nPior caso de matriz de confusão:")
    print(worse_conf)
    
