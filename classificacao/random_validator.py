import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def random_validator(x, y, k, rounds=500):

    acc_list = []
    confusions = []

    for i in range(rounds):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i, shuffle=True
        )
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

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