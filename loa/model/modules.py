import torch
from torch import nn
from torch.nn import functional as F

from loa.model.backbone.modules import init_weight


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=False):
        super(Conv3x3GNReLU, self).__init__()
        self.up_sample = up_sample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, p_channels, skip_channels):
        super(FPNBlock, self).__init__()
        self.skip_conv = nn.Conv2d(skip_channels, p_channels, kernel_size=(1, 1))

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2)
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_up_samples=0):
        super(UpSampleBlock, self).__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, up_sample=bool(num_up_samples))]

        if num_up_samples > 1:
            for i in range(1, num_up_samples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, up_sample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class SpatialGatherModule(nn.Module):
    def __init__(self, cls_num, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        b, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(b, c, -1)
        feats = feats.view(b, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # b x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # b x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True)
        )

        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True)
        )

        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True)
        )

        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(True)
        )
        init_weight(self.modules())

    def forward(self, x, proxy):  # (b, 702, 128, 128), (b, 702, 3, 1)
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # (b, 702, 128, 128) ==> (b, 512, 128, 128) ==> (b, 512, 16384)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)     # (b, 512, 16384)
        query = query.permute(0, 2, 1)                                      # (b, 16384, 512)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)  # (b, 512, 3)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)  # (b, 512, 3)
        value = value.permute(0, 2, 1)                                      # (b, 3, 512)

        sim_map = torch.matmul(query, key)  # (b, 16384, 512) * (b, 512, 3) ==> (b, 16384, 3)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)  # (b, 16384, 3) * (b, 3, 512) ==> (b, 16384, 512)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class SpatialModule(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1):
        super(SpatialModule, self).__init__()
        self.oa_block = _ObjectAttentionBlock(in_channels, key_channels, scale)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        init_weight(self.modules())

    def forward(self, feats, proxy_feats):
        context = self.oa_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, in_channels, (3, 3),
                      padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, (1, 1), groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
          )
        init_weight(self.modules())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, up_sample=False, dropout_rate=0.2):
        super(ASPP, self).__init__()
        self.up_sample = up_sample
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate)
        )
        init_weight(self.modules())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = self.project(torch.cat(res, dim=1))
        if self.up_sample:
            res = F.interpolate(res, scale_factor=2, mode='bilinear', align_corners=True)
        return res
