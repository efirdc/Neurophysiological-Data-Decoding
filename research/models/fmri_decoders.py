from typing import Sequence, Dict, Any, Tuple, Optional
from collections import OrderedDict

import torch
from torch import nn

from .components import Swish
from .components_3d import Block3d, BlurConv3d
from .components_2d import Block2d, BlurConvTranspose2d


class ConvolutionalDecoder(nn.Module):
    """

    """
    def __init__(
            self,
            in_channels: int,
            extractor_channels: Sequence[int],
            decoder_channels: Sequence[int],
            decoder_base_shape: Tuple[int, int, int],
            decoder_output_shapes: Dict[str, Tuple[int,]],
            extractor_block_module: nn.Module = Block3d,
            extractor_block_params: Dict[str, Any] = None,
            extractor_pooling_module: nn.Module = BlurConv3d,
            extractor_pooling_params: Dict[str, Any] = None,
            decoder_block_module: nn.Module = Block2d,
            decoder_block_params: Dict[str, Any] = None,
            decoder_upsampling_module: nn.Module = BlurConvTranspose2d,
            decoder_upsampling_params: Dict[str, Any] = None,
    ):
        super().__init__()

        if extractor_block_params is None:
            extractor_block_params = {}
        if extractor_pooling_params is None:
            extractor_pooling_params = {'stride': 2, 'padding': 1}
        if decoder_block_params is None:
            decoder_block_params = {}
        if decoder_upsampling_params is None:
            decoder_upsampling_params = {'stride': 2, 'padding': 1}

        modules = []
        num_extractor_blocks = len(extractor_channels)
        for i in range(num_extractor_blocks):
            block = extractor_block_module(
                in_channels=in_channels if i == 0 else extractor_channels[i],
                out_channels=extractor_channels[i],
                **extractor_block_params
            )
            modules.append((f'block_{i}', block))
            if i == (num_extractor_blocks - 1):
                break
            pool = extractor_pooling_module(extractor_channels[i], extractor_channels[i + 1], **extractor_pooling_params)
            modules.append((f'pool_{i}', pool))
        modules.append(('global_average_pool', nn.AdaptiveAvgPool3d(1)))
        self.feature_extractor = nn.Sequential(OrderedDict(modules))

        decoder_base_spatial_shape = decoder_base_shape[1:]
        decoder_spatial_shapes = {
            tuple(elem * 2 ** i for elem in decoder_base_spatial_shape): i
            for i in range(len(decoder_channels))
        }
        for output_name, output_shape in decoder_output_shapes.items():
            if len(output_shape) == 1:
                continue
            spatial_output_shape = output_shape[1:]
            error_msg = f'Output {output_name} has unexpected shape {spatial_output_shape}. ' \
                        f'Valid spatial shapes are: {tuple(decoder_spatial_shapes.keys())}'
            if len(output_shape) != 3:
                raise ValueError(error_msg)
            if spatial_output_shape not in decoder_spatial_shapes:
                raise ValueError(error_msg)

        num_decoder_blocks = len(decoder_channels)
        self.decoder_in_conv = nn.ConvTranspose2d(extractor_channels[-1], decoder_channels[0],
                                                  kernel_size=decoder_base_spatial_shape)
        self.decoder_blocks = nn.ModuleList(
            decoder_block_module(in_channels=decoder_channels[i],
                                 out_channels=decoder_channels[i],
                                 **decoder_block_params)
            for i in range(num_decoder_blocks)
        )

        self.decoder_upsampling = nn.ModuleList(
            decoder_upsampling_module(decoder_channels[i], decoder_channels[i + 1], **decoder_upsampling_params)
            for i in range(num_decoder_blocks - 1)
        )

        self.decoder_block_outputs = [[] for block_level in decoder_spatial_shapes.values()]
        self.linear_outputs = {}
        for output_name, output_shape in decoder_output_shapes.items():
            if len(output_shape) == 1:
                self.linear_outputs[output_name] = nn.Linear(extractor_channels[-1], output_shape[0])
                continue
            out_channels, spatial_output_shape = output_shape[0], output_shape[1:]
            block_level = decoder_spatial_shapes[spatial_output_shape]
            in_channels = decoder_channels[block_level]
            output_module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.decoder_block_outputs[block_level].append((output_name, output_module))
        self.linear_outputs = nn.ModuleDict(self.linear_outputs)
        self.decoder_block_outputs = nn.ModuleList([nn.ModuleDict(modules)
                                                    for modules in self.decoder_block_outputs])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x = self.feature_extractor(x)
        y = {}

        for output_name, linear_module in self.linear_outputs.items():
            y[output_name] = linear_module(x.flatten(start_dim=1))

        decoder_features = []
        x = self.decoder_in_conv(x[:, :, 0])
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
            decoder_features.append(x)
            if i == (len(self.decoder_blocks) - 1):
                break
            x = self.decoder_upsampling[i](x)

        for block_level, module_dict in enumerate(self.decoder_block_outputs):
            for output_name, module in module_dict.items():
                y[output_name] = module(decoder_features[block_level])

        return y


class VariationalDecoder(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_size: int,
            decoder_class: Optional[nn.Module] = None,
            decoder_params: Optional[Dict] = None,
    ):

        super().__init__()
        self.out_features = out_features
        if decoder_class is None:
            self.decoder = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_size),
                nn.ReLU(),
                torch.nn.Dropout(p=0.9, inplace=False),
                nn.Linear(hidden_size, out_features * 2),
            )
        else:
            print(decoder_class, decoder_params)
            self.decoder = decoder_class(**decoder_params)

    def forward(self, x):
        distribution = self.decoder(x)
        mu = distribution[:, :self.out_features]
        log_var = distribution[:, self.out_features:]

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = eps * std + mu

        return z, mu, log_var


class SpatialDecoder(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True),
                Swish(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True),
                Swish(),
            )

        def forward(self, x):
            x_in = x
            x = self.block(x)
            return x + x_in

    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_size: int,
            kernel_size: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(in_features, hidden_size, kernel_size),
            nn.GroupNorm(num_groups=32, num_channels=hidden_size, eps=1e-6, affine=True),
            Swish(),
            #self.ResidualBlock(hidden_size),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(hidden_size, out_features, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = x[:, :, None, None]
        x = self.encoder(x)
        return x


class SpatialDiscriminator(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_size: int,
            kernel_size: int,
    ):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_features, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_size, 2, kernel_size=kernel_size),
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.flatten(start_dim=1)
        return x