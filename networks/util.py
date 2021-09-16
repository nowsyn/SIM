import torch
import torch.nn as nn
import functools
import networks.resnet_GN_WS as resnet_GN_WS
import networks.layers_WS as L
import networks.resnet_bn as resnet_bn
from torch.nn import BatchNorm2d
# from modules.nn import BatchNorm2d


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.2, 1.0)
    elif isinstance(m, BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def norm(dim, bn=False):
    if (bn is False):
        return nn.GroupNorm(32, dim)
    else:
        return BatchNorm2d(dim)


class _ConvLayer(nn.Sequential):
    def __init__(self, inc, ouc, kernel_size=1, stride=1, padding=0, bn=True, bias=False, lrelu=False):
        super(_ConvLayer, self).__init__()
        if bn:
            self.add_module('conv', nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)),
            self.add_module('norm', BatchNorm2d(ouc))
            if lrelu:
                self.add_module('relu', nn.LeakyReLU())
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
        else:
            self.add_module('conv', nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)),
            if lrelu:
                self.add_module('relu', nn.LeakyReLU())
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x):
        out = super(_ConvLayer, self).forward(x)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResnetDilatedBN(nn.Module):
    def __init__(self, args, orig_resnet, dilate_scale=8):
        super(ResnetDilatedBN, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.conv1_side = nn.Conv2d(20, 64, 7, 2, 3, bias=False)
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False, smap=None):
        conv_out = [x]
        x = self.relu1(self.bn1(self.conv1(x)+self.conv1_side(smap)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, args, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
  
        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.conv1_side = nn.Conv2d(20, 64, 7, 2, 3, bias=False)
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False, smap=None):
        conv_out = [x]
        x = self.relu(self.bn1(self.conv1(x)+self.conv1_side(smap)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]
