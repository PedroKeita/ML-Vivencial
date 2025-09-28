import numpy as np
from scipy.spatial.distance import cdist

def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
  
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return 0.0


    min_intercluster = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            points_i = X[labels == clusters[i]]
            points_j = X[labels == clusters[j]]
            dist = np.min(cdist(points_i, points_j))
            min_intercluster = min(min_intercluster, dist)

 
    max_intracluster = 0
    for c in clusters:
        points = X[labels == c]
        if len(points) > 1:
            dist = np.max(cdist(points, points))
            max_intracluster = max(max_intracluster, dist)

    return min_intercluster / (max_intracluster + 1e-6)
