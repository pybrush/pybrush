import numpy as np
from sklearn.linear_model import LogisticRegression

class ExactUnlearning:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def unlearn(self, X_forget, y_forget):
        """Remove specific data points and retrain the model."""
        mask = np.ones(len(self.X_train), dtype=bool)
        mask[np.where((self.X_train == X_forget).all(axis=1))[0]] = False
        X_new, y_new = self.X_train[mask], self.y_train[mask]
        
        new_model = LogisticRegression()
        new_model.fit(X_new, y_new)
        self.model = new_model
        return new_model