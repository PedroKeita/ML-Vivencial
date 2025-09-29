"""
Validação do modelo K-NN usando amostragem aleatória.
Calcula estatísticas de acurácia e matrizes de confusão.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def random_validator(X, y, k, rounds=500):

    """
    Executa validação aleatória do modelo K-NN.

    Args:
        X (np.ndarray): Matriz de características N×p
        y (np.ndarray): Vetor de classes N×1
        k (int): Número de vizinhos para K-NN
        rounds (int): Número de rodadas aleatórias

    Returns:
        statistics (dict): Estatísticas de acurácia
        better_conf (tuple): Matriz de confusão do melhor caso
        worse_conf (tuple): Matriz de confusão do pior caso
    """

    acc_list = []
    confusions = []

    for i in range(rounds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, shuffle=True
        )
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        acc_list.append(accuracy)
        confusions.append((accuracy, confusion_matrix(y_test, y_pred)))
    
    statistics = {
        "Média": np.mean(acc_list),
        "Desvio-padrão": np.std(acc_list, ddof=1),
        "Maior valor": np.max(acc_list),
        "Menor valor": np.min(acc_list)
    }

    better_conf = np.argmax(acc_list)
    worse_conf = np.argmin(acc_list)

    return statistics, confusions[better_conf], confusions[worse_conf]