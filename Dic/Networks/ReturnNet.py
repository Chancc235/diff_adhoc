import torch
import torch.nn as nn

class ReturnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(ReturnNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.fc3(x)
        return x
