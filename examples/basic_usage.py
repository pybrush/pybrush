"""
Basic usage of PyBrush for different unlearning techniques.
"""
import numpy as np
from pybrush.unlearning.exact import ExactUnlearning
from pybrush.unlearning.approximate import ApproximateUnlearning
from pybrush.unlearning.amortized import AmortizedUnlearning
from pybrush.unlearning.certified import CertifiedUnlearning
from pybrush.models.base import PyBrushModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import torch

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Exact Unlearning Example
X_forget, y_forget = X[:10], y[:10]
exact_unlearning = ExactUnlearning(model, X, y)
new_model = exact_unlearning.unlearn(X_forget, y_forget)

# Approximate Unlearning Example
torch_model = PyBrushModel(20, 2)
approx_unlearning = ApproximateUnlearning(torch_model)
torch_X_forget = torch.rand((5, 20))
torch_y_forget = torch.randint(0, 2, (5,))
approx_unlearning.unlearn(torch_X_forget, torch_y_forget)

# Amortized Unlearning Example
amortized_unlearning = AmortizedUnlearning(model)
amortized_unlearning.unlearn(X_forget, y_forget)

# Certified Unlearning Example
certified_unlearning = CertifiedUnlearning(model)
certified_unlearning.unlearn(X_forget, y_forget)