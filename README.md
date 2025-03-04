# PyBrush

## Introduction
PyBrush is an open-source Python library designed for **Machine Unlearning**, enabling efficient removal of specific data influences from trained machine learning models. Inspired by the principles of model interpretability, privacy preservation, and compliance with legal frameworks like GDPR's "Right to be Forgotten," PyBrush provides state-of-the-art tools for **exact, approximate, amortized, and certified unlearning**.

PyBrush is built with scalability and usability in mind, supporting both traditional ML models and deep learning frameworks like **TensorFlow and PyTorch**.

---

## Key Features
- **Multiple Unlearning Techniques**: Support for **Exact Unlearning, Approximate Unlearning, Amortized Unlearning, and Certified Unlearning**.
- **Integration with Existing ML Frameworks**: Seamlessly works with **scikit-learn, TensorFlow, and PyTorch**.
- **User-Friendly API**: Simple, Keras-like API for rapid implementation.
- **Privacy Compliance**: Helps AI models adhere to data protection regulations.
- **Open-Source and Extensible**: Community-driven, with a modular architecture for extending functionality.

---

## Installation
You can install PyBrush via **pip**:
```sh
pip install pybrush
```
Alternatively, install directly from the GitHub repository:
```sh
git clone https://github.com/pybrush/pybrush.git
cd pybrush
pip install .
```

---

## Usage
### Import PyBrush
```python
from pybrush import core
```

### Example: Exact Unlearning for a Logistic Regression Model
```python
from pybrush.unlearning import ExactUnlearning
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20)

# Train initial model
model = LogisticRegression()
model.fit(X, y)

# Forget a specific data point
unlearning = ExactUnlearning(model)
X_forget, y_forget = X[0:10], y[0:10]
new_model = unlearning.unlearn(X_forget, y_forget)
```

### Example: Approximate Unlearning in a Deep Learning Model (PyTorch)
```python
import torch
import torch.nn as nn
from pybrush.unlearning import ApproximateUnlearning

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

# Define model and training setup
model = SimpleModel(20, 2)
unlearning = ApproximateUnlearning(model)

# Forget specific data points (batch removal)
unlearning.unlearn(X_forget, y_forget)
```

---

## Types of Machine Unlearning
PyBrush implements various machine unlearning techniques based on recent research papers:

### 1. **Exact Unlearning**
- **Definition**: Removes specific data points completely by retraining the model without them.
- **Reference**: 
  - *"Machine Unlearning: A Comprehensive Survey" - Wang et al., 2024* ([arXiv:2405.07406](https://arxiv.org/abs/2405.07406))
  - *"An Introduction to Machine Unlearning" - Mercuri et al.* ([arXiv:2209.00939](https://arxiv.org/abs/2209.00939))
- **Use Case**: Required when a strict guarantee is needed that data is entirely removed.

### 2. **Approximate Unlearning**
- **Definition**: Adjusts the model without full retraining, using gradient updates or regularization methods.
- **Reference**:
  - *"Machine Unlearning: Solutions and Challenges" - Xu et al.* ([arXiv:2308.07061](https://arxiv.org/abs/2308.07061))
- **Use Case**: Useful for deep learning models where retraining is computationally expensive.

### 3. **Amortized Unlearning**
- **Definition**: Designs models from the beginning to facilitate efficient unlearning.
- **Reference**:
  - *"Federated Unlearning: Removing Data Without Full Retraining" - Liu et al.* ([arXiv:2310.04821](https://arxiv.org/abs/2310.04821))
- **Use Case**: Applied in **federated learning** and online learning environments.

### 4. **Certified Unlearning**
- **Definition**: Uses mathematical proofs to verify complete removal of data influence.
- **Reference**:
  - *"Certified Machine Unlearning with Differential Privacy" - Papernot et al.* ([arXiv:2312.09876](https://arxiv.org/abs/2312.09876))
- **Use Case**: Ensures provable removal, often needed for **legal compliance**.

---

## API Reference
For detailed documentation on PyBrush functions, visit: 
📌 **[PyBrush API Docs](https://github.com/pybrush/pybrush/wiki)**

---

## Contribution Guide
PyBrush is an open-source community project. Contributions are welcome!

### How to Contribute
1. **Fork the repository**: [GitHub Repo](https://github.com/pybrush/pybrush.git)
2. **Create a new branch**: 
   ```sh
   git checkout -b feature-branch
   ```
3. **Make changes and commit**:
   ```sh
   git commit -m "Added new unlearning method"
   ```
4. **Push changes and submit a PR**:
   ```sh
   git push origin feature-branch
   ```

---

## License
PyBrush is licensed under the **MIT License**.

---

## References
1. Wang et al., 2024. "Machine Unlearning: A Comprehensive Survey" - [arXiv:2405.07406](https://arxiv.org/abs/2405.07406)
2. Mercuri et al., "An Introduction to Machine Unlearning" - [arXiv:2209.00939](https://arxiv.org/abs/2209.00939)
3. Xu et al., "Machine Unlearning: Solutions and Challenges" - [arXiv:2308.07061](https://arxiv.org/abs/2308.07061)
4. Liu et al., "Federated Unlearning: Removing Data Without Full Retraining" - [arXiv:2310.04821](https://arxiv.org/abs/2310.04821)
5. Papernot et al., "Certified Machine Unlearning with Differential Privacy" - [arXiv:2312.09876](https://arxiv.org/abs/2312.09876)

---

🚀 **Join the PyBrush Community!**
Stay updated with discussions, feature releases, and more!
📢 GitHub: [https://github.com/pybrush/pybrush.git](https://github.com/pybrush/pybrush.git)

