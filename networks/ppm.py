import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.layers_WS as L
from networks.util import norm 


class PPM(nn.Module):
    def __init__(self, in_channels, pool_scales, out_channels=256, batch_norm=True):
        super().__init__()
        # ppm module
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                L.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                norm(out_channels, batch_norm),
                nn.LeakyReLU()
            ))
        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, inp):
        input_size = inp.size()
        ppm_out = []
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(inp), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        return ppm_out

         
# ASPP module
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, batch_norm=True):
        modules = [
            L.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            norm(out_channels, batch_norm),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            L.Conv2d(in_channels, out_channels, 1, bias=False),
            norm(out_channels, batch_norm),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, batch_norm=True):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            L.Conv2d(in_channels, out_channels, 1, bias=False),
            norm(out_channels, batch_norm),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, batch_norm))

        modules.append(ASPPPooling(in_channels, out_channels, batch_norm))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            L.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            norm(out_channels, batch_norm),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
