from collections.abc import Sequence

import torch
from torch import nn


class DenseLayer(nn.Module):
    """BN-ReLU-Conv1x1 bottleneck and BN-ReLU-Conv adding `growth_rate` channels to its input."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        kernel_size: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        bottleneck_channels = bn_size * growth_rate
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                bottleneck_channels,
                growth_rate,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layers(x)], dim=1)


class Transition(nn.Sequential):
    """BN-ReLU-Conv1x1 shrinking the channels between dense blocks, then halving the length."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )


class DenseNet1d(nn.Module):
    """DenseNet-BC (https://arxiv.org/abs/1608.06993) with 1D convolutions along the signal.

    Maps signals (batch, in_channels, samples) to logits (batch, output_size). The stem and the
    transitions shorten the signal 2 ** (len(block_config) + 1) times before global pooling.
    The defaults are DenseNet-121.
    """

    def __init__(
        self,
        in_channels: int,
        output_size: int,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        kernel_size: int = 7,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to keep the signal length, got {kernel_size}"
            )
        if not 0 < compression <= 1:
            raise ValueError(f"compression must be in (0, 1], got {compression}")

        self.features = nn.Sequential()
        self.features.add_module(
            "stem",
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm1d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ),
        )

        channels = num_init_features
        for i, num_layers in enumerate(block_config, start=1):
            block = nn.Sequential(
                *(
                    DenseLayer(
                        channels + j * growth_rate,
                        growth_rate,
                        bn_size,
                        kernel_size,
                        drop_rate,
                    )
                    for j in range(num_layers)
                )
            )
            self.features.add_module(f"block{i}", block)
            channels += num_layers * growth_rate

            if i < len(block_config):
                out_channels = int(channels * compression)
                self.features.add_module(
                    f"transition{i}", Transition(channels, out_channels)
                )
                channels = out_channels

        self.head = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, output_size),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
