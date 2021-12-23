from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F


class Block1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_convs: int = 2,
            conv_class: nn.Module = nn.Conv1d,
            conv_params: Optional[Dict[str, Any]] = None,
            normalization_class: nn.Module = nn.BatchNorm1d,
            normalization_params: Optional[Dict[str, Any]] = None,
            activation_class: nn.Module = nn.ReLU,
            activation_params: Optional[Dict[str, Any]] = None,
            residual: bool = False,
            residual_params: Optional[Dict[str, Any]] = None,
            dropout_p: Optional[float] = None,
    ):
        super().__init__()

        if conv_params is None:
            conv_params = {'bias': False, 'kernel_size': 3, 'padding': 1}
        if normalization_params is None:
            normalization_params = {}
        if activation_params is None:
            activation_params = {'inplace': True}
        if residual_params is None:
            residual_params = {'bias': True, 'kernel_size': 3, 'padding': 1}

        self.residual = residual
        if self.residual:
            self.res_conv = conv_class(in_channels, out_channels, **residual_params)

        parts = []
        for i in range(num_convs):
            layer_channels = in_channels if i == 0 else out_channels
            parts.append((f'conv{i}', conv_class(layer_channels, out_channels, **conv_params)))
            parts.append((f'norm{i}', normalization_class(out_channels, **normalization_params)))
            parts.append((f'activation{i}', activation_class(**activation_params)))
        self.layers = nn.Sequential(OrderedDict(parts))

        self.dropout = None
        if dropout_p:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x_in = x

        x = self.layers(x)

        if self.residual:
            x = self.res_conv(x_in) + x

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class BlurConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        kernel = torch.ones(in_channels, 1, 2)
        kernel = kernel / 2
        self.register_buffer('kernel', kernel)

        # Signal is shrinking by stride
        self.kernel = self.kernel / self.stride[0]
        self.kwargs = kwargs

    def forward(self, x):
        weight = self.weight

        weight = F.conv1d(weight, self.kernel, padding=1, groups=self.in_channels)
        x = F.conv1d(x, weight, **self.kwargs)

        return x


class BlurConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        kernel = torch.ones(out_channels, 1, 2)
        kernel = kernel / 2
        self.register_buffer('kernel', kernel)

        # Signal is growing by stride
        self.kernel = self.kernel * self.stride
        self.kwargs = kwargs

    def forward(self, x, output_size=None):
        weight = self.weight

        weight = F.conv1d(weight, self.kernel, padding=1, groups=self.in_channels)
        x = F.conv_transpose1d(x, weight, **self.kwargs)

        return x
