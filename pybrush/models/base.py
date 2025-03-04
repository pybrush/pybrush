import torch.nn as nn
import torch.optim as optim

class PyBrushModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyBrushModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

    def train_model(self, data, labels):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = loss_fn(self.forward(data), labels)
        loss.backward()
        optimizer.step()