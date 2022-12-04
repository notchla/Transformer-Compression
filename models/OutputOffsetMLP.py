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
        layers = nn.ModuleList([first_layer])

        for _ in range(self.depth -2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.output_size))
        self.layers = layers
        self.layers.apply(self._init_weights)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.

    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(self, latent_dim, num_modulations, dim_hidden, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), nn.ReLU()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)

class ModulatedPointWiseOutputOffsetMLP(nn.Module):

    def __init__(self, depth = 4, hidden_dim = 32, token_size = 64, act_layer = nn.ReLU(), modulation_net_dim_hidden=64, modulation_net_num_layers=1):
        super().__init__()
        self.depth = depth
        self.input_size = token_size
        self.hidden_dim = hidden_dim
        self.output_size = 3
        self.activation = act_layer

        num_modulations = hidden_dim * (depth-1)
        self.modulation_net = LatentToModulation(
                token_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        
        self.num_hidden_layers = depth-1

        first_layer = nn.Linear(self.input_size, self.hidden_dim)
        layers = nn.ModuleList([first_layer])

        for _ in range(self.depth -2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.output_size))
        self.layers = layers
        self.layers.apply(self._init_weights)

    def forward(self, x, cls_token):

        modulations = self.modulation_net(cls_token)

        idx = 0
        for layer_idx in range(self.num_hidden_layers):
            layer = self.layers[layer_idx]
            x = layer(x)
            shift = modulations[:, idx:idx+self.hidden_dim].unsqueeze(1)
            x = x + shift
            x = self.activation(x)
            idx = idx + self.hidden_dim
        
        out = self.layers[-1](x)
        # x = self.activation(x)

        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)