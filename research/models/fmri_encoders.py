from typing import Sequence, Dict, Any, Tuple, Optional
from collections import OrderedDict
import math

import torch
from torch import nn
from einops import rearrange

from .components import Swish
from .components_3d import Block3d, BlurConv3d
from .components_2d import Block2d, BlurConvTranspose2d


class Encoder(nn.Module):
    def __init__(
            self,
            layer_sizes: Sequence[int],
            dropout_p: Optional[float] = None,
    ):
        super().__init__()
        self.layers = [nn.Linear(in_features=layer_sizes[0], out_features=layer_sizes[1])]
        for in_size, out_size in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.layers += [
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_size, out_size)
            ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x


class SpatialEncoder(nn.Module):
    def __init__(
            self,
            input_shape: Sequence[int],
            output_size: int,
            channels_last: bool = False,
    ):
        super().__init__()
        self.channels_last = channels_last
        if channels_last:
            H, W, C = input_shape
        else:
            C, H, W = input_shape
        V = output_size

        #self.linear1 = nn.Linear(H * W, V)
        #self.linear2 = nn.Linear(C, V)
        self.weights1 = nn.Parameter(torch.randn(size=(H, W, V)) / math.sqrt(H * W))
        #self.bias1 = nn.Parameter(torch.zeros(size=(C, V)))
        self.weights2 = nn.Parameter(torch.randn(size=(C, V)) / math.sqrt(C))
        self.bias2 = nn.Parameter(torch.zeros(size=(V,)))

    def forward(self, x):
        in_shape = 'nhwc' if self.channels_last else 'nchw'
        x = torch.einsum(f'{in_shape}, hwv -> ncv', x, self.weights1)
        x = torch.einsum('ncv, cv -> nv', x, self.weights2) + self.bias2

        #x = torch.einsum('ncd, vd -> ncv', x.flatten(start_dim=2), self.linear1.weight) + self.linear1.bias
        #x = torch.einsum('ncv, vc -> nv', x, self.linear2.weight) + self.linear2.bias
        return x
