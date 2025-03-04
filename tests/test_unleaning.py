import unittest
import numpy as np
from pybrush.unlearning.exact import ExactUnlearning
from pybrush.unlearning.approximate import ApproximateUnlearning
from pybrush.models.base import PyBrushModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import torch

class TestUnlearning(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=20)
        self.model = LogisticRegression()
        self.model.fit(self.X, self.y)
        self.X_forget, self.y_forget = self.X[:5], self.y[:5]

    def test_exact_unlearning(self):
        unlearning = ExactUnlearning(self.model, self.X, self.y)
        new_model = unlearning.unlearn(self.X_forget, self.y_forget)
        self.assertTrue(new_model.score(self.X_forget, self.y_forget) < self.model.score(self.X_forget, self.y_forget))

    def test_approximate_unlearning(self):
        model = PyBrushModel(20, 2)
        X_forget = torch.rand((5, 20))
        y_forget = torch.randint(0, 2, (5,))
        unlearning = ApproximateUnlearning(model)
        new_model = unlearning.unlearn(X_forget, y_forget)
        self.assertIsNotNone(new_model)

if __name__ == "__main__":
    unittest.main()