import math
from typing import Tuple

import torch
import torch.nn as nn

from .modules import LayerNorm, init_weight


class _LDCBlock(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, dropout_rate: float):
        super(_LDCBlock, self).__init__()

        self.dws_conv_7x7 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=(7, 7), padding=3, groups=n_channels, bias=False),
            LayerNorm(n_channels),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )

        self.dw_conv_3x3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=1, groups=n_channels, bias=False),
            LayerNorm(n_channels),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )

        self.proj = nn.Sequential(
            nn.Conv2d(n_channels, growth_rate, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(growth_rate)
        )
        init_weight(self.modules())

    def forward(self, x):
        out = x + self.dws_conv_7x7(x) + self.dw_conv_3x3(x) + self.conv_1x1(x)
        out = self.proj(out)
        out = torch.cat((x, out), dim=1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, dropout_rate: float):
        super(_Transition, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(n_channels, n_out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(n_out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout_rate / 2),
            nn.AvgPool2d(kernel_size=2, ceil_mode=True)
        )
        init_weight(self.modules())

    def forward(self, x):
        return self.transition(x)


class LDCNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            growth_rate: int = 24,
            num_layers: Tuple[int, int, int, int] = (2, 2, 6, 2),
            reduction: float = 0.5,
            dropout_rate: float = 0.2
    ):
        super(LDCNet, self).__init__()
        # input
        n_channels = 2 * growth_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=(1, 1), bias=False),
            nn.Conv2d(n_channels, n_channels, kernel_size=(7, 7), padding=3, stride=(2, 2),
                      groups=n_channels, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # first dense
        n_dense_blocks = num_layers[0]
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, dropout_rate)
        n_channels += n_dense_blocks * growth_rate

        # first dense out channels
        self.channels1 = n_channels

        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, dropout_rate)

        # second dense
        n_channels = n_out_channels
        n_dense_blocks = num_layers[1]
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, dropout_rate)
        n_channels += n_dense_blocks * growth_rate

        # second dense out channels
        self.channels2 = n_channels

        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, dropout_rate)

        # third dense
        n_channels = n_out_channels
        n_dense_blocks = num_layers[2]
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, dropout_rate)
        n_channels += n_dense_blocks * growth_rate

        # third dense out channels
        self.channels3 = n_channels

        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans3 = _Transition(n_channels, n_out_channels, dropout_rate)

        # forth dense
        n_channels = n_out_channels
        n_dense_blocks = num_layers[3]
        self.dense4 = self._make_dense(n_channels, growth_rate, n_dense_blocks, dropout_rate)

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

        # out channels
        self.channels4 = self.out_channels

        init_weight(self.modules())

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, dropout_rate):
        layers = []
        for _ in range(int(n_dense_blocks)):
            layers.append(_LDCBlock(n_channels, growth_rate, dropout_rate))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def get_channels(self):
        return self.channels4, self.channels3, self.channels2, self.channels1

    def forward(self, x):
        out = self.stem(x)

        out3 = self.dense1(out)  # (240, 128, 128)

        out = self.trans1(out3)

        out2 = self.dense2(out)  # (312, 64, 64)

        out = self.trans2(out2)
        out1 = self.dense3(out)  # (732, 32, 32)

        out = self.trans3(out1)
        out = self.dense4(out)
        out = self.post_norm(out)  # (558, 16, 16)
        return out3, out2, out1, out
