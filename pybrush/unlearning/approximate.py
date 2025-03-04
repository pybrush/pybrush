import torch
import torch.nn as nn
import torch.optim as optim

class ApproximateUnlearning:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

    def unlearn(self, X_forget, y_forget):
        """Use gradient updates to approximate unlearning."""
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model.forward(X_forget), y_forget)
        loss.backward()
        self.optimizer.step()
        return self.model