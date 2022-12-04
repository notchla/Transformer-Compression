import torch
import torch.nn as nn
import math
import numpy as np

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None, scale=2):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.sidelength = sidelength
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies
        # self.frequencies_per_axis = (num_frequencies * np.array(sidelength)) // max(sidelength)
        self.out_dim = in_features + in_features * 2 * self.num_frequencies  # (sum(self.frequencies_per_axis))

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        # coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):

            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((self.scale ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((self.scale ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc

def get_positional_encoding_layer(args=None, img_resolution=None, in_features=2):
    """
    Gets layer for positional encoding

    Args:
        encoding: str. Type of positional encoding
        ff_dims: int. Number of frequencies used for positional encoding
        in_features: int. Dimensionality of the input
        kwargs
    Returns:
        nn.Module or None
    """


    positional_encoding = PosEncodingNeRF(
        in_features=in_features,
        sidelength=img_resolution,
        fn_samples=None,
        use_nyquist=True,
        num_frequencies=args.ff_dims,
        scale=args.encoding_scale)

    return positional_encoding

class PointWisePositionMLP(nn.Module):

    def __init__(self, depth = 4, hidden_dim = 32, token_size = 32, act_layer = nn.ReLU(), args=None, img_resolution=None):
        super().__init__()

        in_features = 2
        self.positional_encoding = get_positional_encoding_layer(
            args=args, img_resolution=img_resolution) if args.pos else None
        if self.positional_encoding:
            in_features = self.positional_encoding.out_dim

        self.depth = depth
        self.input_size = in_features
        self.hidden_dim = hidden_dim
        self.token_size = token_size
        self.activation = act_layer

        first_layer = nn.Linear(self.input_size, self.hidden_dim)
        layers = nn.ModuleList([first_layer])

        for _ in range(self.depth -2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.token_size))
        self.layers = layers
        self.layers.apply(self._init_weights)

            

    def forward(self, x):

        if self.positional_encoding:
            x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)



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


