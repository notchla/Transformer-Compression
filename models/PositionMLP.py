import torch
import torch.nn as nn

class PointWisePositionMLP(nn.Module):

    def __init__(self, depth = 4, hidden_dim = 32, token_size = 32, act_layer = nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.input_size = 2
        self.hidden_dim = hidden_dim
        self.token_size = token_size
        self.activation = act_layer

        first_layer = nn.Linear(self.input_size, self.hidden_dim)
        layers = [first_layer]

        for _ in range(self.depth -2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.token_size))
        self.layers = layers

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return x



class GlobalPositionMLP(nn.Module):

    def __init__(self, res, depth = 5, hidden_dim = 32, token_size = 32, act_layer = nn.ReLU()):
        super().__init__()
        self.res = res
        self.depth = depth
        self.input_size = self.res * 2
        self.hidden_dim = self.res * hidden_dim
        self.token_size = self.res * token_size
        self.activation = act_layer

        first_layer = nn.Linear(self.input_size, self.hidden_dim)
        layers = [first_layer]

        for _ in range(self.depth -2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.token_size))
        self.layers = layers

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return x


