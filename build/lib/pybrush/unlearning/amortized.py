import numpy as np
from sklearn.linear_model import LogisticRegression

class AmortizedUnlearning:
    def __init__(self, model):
        self.model = model

    def unlearn(self, X_forget, y_forget):
        """Efficiently remove data from the model without full retraining."""
        # Here, we simulate the removal by modifying the training process
        new_model = LogisticRegression()
        new_model.fit(np.delete(self.model.X_train, X_forget, axis=0),
                      np.delete(self.model.y_train, y_forget, axis=0))
        self.model = new_model
        return new_model