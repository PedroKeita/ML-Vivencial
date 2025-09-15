import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score


def choose_k(x, y, k_values):
    results = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, x, y, cv=kf)
        results[k] = scores.mean()
    
    better_k = max(results, key=results.get)
    return better_k, results