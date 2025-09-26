from sklearn.preprocessing import StandardScaler
import numpy as np

def normalize(X: np.ndarray) -> np.ndarray:

    scaler = StandardScaler()
    return scaler.fit_transform(X)