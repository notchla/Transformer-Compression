import torch
import torch.nn as nn

class PointWiseOutputOffsetMLP(nn.Module):

    def __init__(self, depth = 4, hidden_dim = 32, token_size = 64, act_layer = nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.input_size = token_size
        self.hidden_dim = hidden_dim
        self.output_size = 3
        self.activation = act_layer

        first_layer = nn.Linear(self.input_size, self.hidden_dim)
        layers = [first_layer]

        for _ in range(self.depth -2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.output_size))
        self.layers = layers

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return x