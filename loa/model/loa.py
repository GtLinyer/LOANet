from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import LDCNet
from .modules import init_weight, ASPP, SpatialGatherModule, SpatialModule, FPNBlock, UpSampleBlock


class LOANet(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels: int = 3,
            growth_rate: int = 24,
            num_layers: Tuple[int, int, int, int] = (2, 2, 6, 2),
            reduction: float = 0.5,
            p_channels: int = 64,
            u_channels: int = 64,
            attention_mid_channels: int = 384,
            attention_scale: int = 8,
            dropout_rate: float = 0.2
    ):
        super(LOANet, self).__init__()
        self.backbone = LDCNet(in_channels, growth_rate, num_layers, reduction, dropout_rate)
        backbone_channels = self.backbone.get_channels()

        self.aspp1 = ASPP(backbone_channels[3], p_channels, atrous_rates=(12, 24, 36), dropout_rate=dropout_rate)
        self.aspp2 = ASPP(backbone_channels[2], p_channels, atrous_rates=(12, 24, 36), dropout_rate=dropout_rate)
        self.aspp3 = ASPP(backbone_channels[1], p_channels, atrous_rates=(6, 12, 18), dropout_rate=dropout_rate)
        self.conv4 = nn.Conv2d(backbone_channels[0], p_channels, kernel_size=(1, 1))

        self.fpn3 = FPNBlock(p_channels, p_channels)
        self.fpn2 = FPNBlock(p_channels, p_channels)
        self.fpn1 = FPNBlock(p_channels, p_channels)

        self.up_sample4 = UpSampleBlock(p_channels, u_channels, num_up_samples=3)
        self.up_sample3 = UpSampleBlock(p_channels, u_channels, num_up_samples=2)
        self.up_sample2 = UpSampleBlock(p_channels, u_channels, num_up_samples=1)
        self.up_sample1 = UpSampleBlock(p_channels, u_channels, num_up_samples=0)

        o_channels = u_channels * 4
        self.aux = nn.Sequential(
            nn.Conv2d(o_channels, o_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(o_channels),
            nn.ReLU(True),
            nn.Conv2d(o_channels, num_classes, kernel_size=(1, 1))
        )
        self.sg = SpatialGatherModule(num_classes, 2)
        self.s_oa = SpatialModule(o_channels, attention_mid_channels, o_channels, attention_scale)

        self.classifier = nn.Sequential(
            nn.Conv2d(o_channels, o_channels, (3, 3), padding=1, groups=o_channels, bias=False),
            nn.Conv2d(o_channels, num_classes, (1, 1), bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(True),
            nn.Conv2d(num_classes, num_classes, kernel_size=(1, 1))
        )
        init_weight(self.modules())

    def forward(self, x):
        # backbone
        out1, out2, out3, out4 = self.backbone(x)

        out1 = self.aspp1(out1)
        out2 = self.aspp2(out2)
        out3 = self.aspp3(out3)
        f4 = self.conv4(out4)
        f3 = self.fpn3(f4, out3)
        f2 = self.fpn2(f3, out2)
        f1 = self.fpn1(f2, out1)

        sc4 = self.up_sample4(f4)
        sc3 = self.up_sample3(f3)
        sc2 = self.up_sample2(f2)
        sc1 = self.up_sample1(f1)

        out = torch.cat([sc4, sc3, sc2, sc1], dim=1)

        out_aux = self.aux(out)
        context = self.sg(out, out_aux)
        out = self.s_oa(out, context)

        out = self.classifier(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out_aux = F.interpolate(out_aux, scale_factor=4, mode='bilinear', align_corners=True)
        if self.training:
            return out, out_aux
        else:
            return out
