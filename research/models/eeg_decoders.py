from typing import Sequence, Dict, Any
from collections import OrderedDict

from torch import nn

from .components_1d import Block1d, BlurConv1d


class ConvolutionalDecoder(nn.Module):
    """
    1 dimensional decoder network
    Applies a sequence of convolution and pooling operations, and then goes fully connected
    """
    def __init__(
            self,
            in_samples: int,
            in_channels: int,
            block_channels: Sequence[int],
            fully_connected_features: Sequence[int],
            block_module: nn.Module = Block1d,
            block_params: Dict[str, Any] = None,
            pooling_module: nn.Module = BlurConv1d,
            pooling_params: Dict[str, Any] = None,
    ):
        super().__init__()

        if block_params is None:
            block_params = {}
        if pooling_params is None:
            pooling_params = {'stride': 2, 'padding': 1}

        num_blocks = len(block_channels)
        self.blocks = nn.ModuleList(
            block_module(
                in_channels=in_channels if i == 0 else block_channels[i],
                out_channels=block_channels[i],
                **block_params
            )
            for i in range(num_blocks)
        )

        self.pooling = nn.ModuleList(
            pooling_module(block_out, block_in, **pooling_params)
            for block_out, block_in in zip(block_channels[:-1], block_channels[1:])
        )

        out_samples = int(in_samples / (2 ** (num_blocks - 1)))
        fully_connected_features_in = out_samples * block_channels[-1]
        fully_connected_layers = [
            ('fc0', nn.Linear(in_features=fully_connected_features_in, out_features=fully_connected_features[0]))
        ]
        for i in range(len(fully_connected_features) - 1):
            fully_connected_layers += [
                (f"activation{i}", nn.ReLU(inplace=True)),
                (f"fc{i + 1}", nn.Linear(fully_connected_features[i], fully_connected_features[i + 1])),
            ]
        self.fully_connected_layers = nn.Sequential(OrderedDict(fully_connected_layers))

    def forward(self, x):

        x = self.blocks[0](x)

        for pool, block in zip(self.pooling, self.blocks[1:]):
            x = pool(x)
            x = block(x)

        N, C, T = x.shape
        x = x.reshape(N, C * T)

        x = self.fully_connected_layers(x)

        return x


class MLPDecoder(nn.Module):
    """
    1 dimensional decoder network
    Applies a sequence of convolution and pooling operations, and then goes fully connected
    """
    def __init__(
            self,
            features: Sequence[int]
    ):
        super().__init__()

        self.layers = [
            nn.Flatten(),
            nn.Linear(features[0], features[1])
        ]
        for i in range(1, len(features) - 1):
            self.layers += [
                nn.ReLU(),
                nn.Linear(features[i], features[i + 1])
            ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    import torch
    device = torch.device('cuda')
    model = ConvolutionalDecoder(
        in_samples=200,
        in_channels=63,
        block_channels=[256, 512, 1024, 2048],
        fully_connected_features=[1024, 512, 120]
    )

    x = torch.randn(16, 63, 200).to(device)
    model = model.to(device)

    z = model(x)
    print(z.shape)
