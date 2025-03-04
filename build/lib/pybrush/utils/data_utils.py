import numpy as np

def preprocess_data(X, y):
    """Preprocess input data for PyBrush models."""
    return np.array(X), np.array(y)