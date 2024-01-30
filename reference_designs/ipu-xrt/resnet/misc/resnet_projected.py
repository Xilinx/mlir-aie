import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim

import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.stats import StatsOp

from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# class Bottlenecklock(nn.Module):
#     """(convolution => [BN] => ReLU)"""

#     def __init__(self, in_channels,out_channels,kernel_size=3,groups=1,
#         padding=1, reduction: int = 4)-> None:
#         super().__init__()

#         mid_channels = out_channels // reduction

#         self.bot_conv1=nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False)
#         self.bot_bn1=nn.BatchNorm2d(mid_channels)

#         self.bot_conv2=nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False)
#         self.bot_bn2=nn.BatchNorm2d(mid_channels)

#         self.bot_conv3=nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False)
#         self.bot_bn3=nn.BatchNorm2d(out_channels)

#         self.bot_relu=nn.ReLU(inplace=True)


#     def forward(self, x):
#         identity = x
#         x=self.bot_conv1(x)
#         x=self.bot_bn1(x)
#         x=self.bot_relu(x)

#         x=self.bot_conv2(x)
#         x=self.bot_bn2(x)
#         x=self.bot_relu(x)

#         x=self.bot_conv3(x)
#         x=self.bot_bn3(x)

#         x=x+ identity
#         x=self.bot_relu(x)
#         return x


class QuantBottleneck_projected_OFFLOAD_FUSED(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, option="A", padding_mode="replicate"
    ):
        super(QuantBottleneck_projected_OFFLOAD_FUSED, self).__init__()
        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )

        if padding_mode == "replicate":
            self.conv2 = QuantConv2d(
                planes,
                planes,
                kernel_size=3,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                padding=1,
                padding_mode=padding_mode,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
        else:
            self.conv2 = QuantConv2d(
                planes,
                planes,
                kernel_size=3,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                padding=1,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )

        self.conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.relu1 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.relu2 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.relu3 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        def forward(self, x):
            out = self.quant_id_1(x)
            out = self.conv1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.conv3(out)
            out = self.quant_id_1(out)
            out += x
            out = self.relu3(out)
            return out


class BasicBlock(nn.Module):
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class QuantBottleneck_projected(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A", padding_mode="zeros"):
        super(QuantBottleneck_projected, self).__init__()

        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_id_2 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_id_3 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_id_4 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        if padding_mode == "replicate":
            self.conv2 = QuantConv2d(
                planes,
                planes,
                kernel_size=3,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                padding=1,
                padding_mode=padding_mode,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
        else:
            self.conv2 = QuantConv2d(
                planes,
                planes,
                kernel_size=3,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                padding=1,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
        self.conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.relu1 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.relu2 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.relu3 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    bias_quant=None,
                    return_quant_tensor=False,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.quant_id_1(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class QuantBottleneck_projected_OFFLOAD(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, option="A", padding_mode="replicate"
    ):
        super(QuantBottleneck_projected_OFFLOAD, self).__init__()
        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_id_2 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_id_3 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_id_4 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        if padding_mode == "replicate":
            self.conv2 = QuantConv2d(
                planes,
                planes,
                kernel_size=3,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                padding=1,
                padding_mode=padding_mode,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
        else:
            self.conv2 = QuantConv2d(
                planes,
                planes,
                kernel_size=3,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                padding=1,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )

        self.conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.relu1 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.relu2 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.relu3 = QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    # def forward(self, x):
    #     out = self.quant_id_1(x)
    #     out = self.relu1(self.bn1(self.conv1(out)))
    #     out = self.relu2(self.bn2(self.conv2(out)))
    #     out = self.bn3(self.conv3(out))
    #     out += x
    #     out = self.relu3(out)
    #     return out

    def forward(self, x):
        out = self.quant_id_1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.quant_id_1(out)
        out += x
        out = self.relu3(out)
        return out


class Bottleneck_projected_OFFLOAD(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, option="A", padding_mode="replicate"
    ):
        super(Bottleneck_projected_OFFLOAD, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if padding_mode == "replicate":
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                padding_mode="replicate",
                stride=stride,
                padding=1,
                bias=False,
            )
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = self.relu3(out)
        return out


class QuantResNet_projected(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(QuantResNet_projected, self).__init__()
        self.in_planes = 64
        self.quant_inp = qnn.QuantIdentity(bit_width=8)

        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=8
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = qnn.QuantLinear(
            512 * block.expansion,
            num_classes,
            weight_bit_width=8,
            bias_quant=None,
            return_quant_tensor=False,
            bias=False,
        )
        self.relu = qnn.QuantReLU(bit_width=8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_inp(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_projected(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_projected, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_projected_IMAGENET(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_projected_IMAGENET, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class SingleBottleneckAIE(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(SingleBottleneckAIE, self).__init__()
        self.in_planes = 64

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(out)
        out = F.avg_pool2d(out, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TripleBottleneck(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(TripleBottleneck, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DoubleBottleneck(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(DoubleBottleneck, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 16)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out


class QuantNonAIELayerInitial(nn.Module):
    def __init__(self):
        super(QuantNonAIELayerInitial, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=8
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = qnn.QuantReLU(bit_width=8)

    def forward(self, x):
        x = self.quant_inp(x)
        out = self.conv1(x)
        out = self.bn1(out)
        # print( out)
        out = self.relu1(out)
        return out


class QuantNonAIELayerPost(nn.Module):
    def __init__(self, num_classes):
        super(QuantNonAIELayerPost, self).__init__()
        expansion = 4
        self.linear = qnn.QuantLinear(
            64 * expansion,
            num_classes,
            weight_bit_width=8,
            bias_quant=None,
            return_quant_tensor=False,
            bias=False,
        )

    def forward(self, x):
        out = F.avg_pool2d(x, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class NonAIELayerInitial32(nn.Module):
    def __init__(self):
        super(NonAIELayerInitial32, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # print( out)
        out = F.relu(out)
        return out


class NonAIELayerInitial(nn.Module):
    def __init__(self):
        super(NonAIELayerInitial, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # print( out)
        out = F.relu(out)
        return out


class DualConvNonAIELayerPost(nn.Module):
    def __init__(self, num_classes):
        super(DualConvNonAIELayerPost, self).__init__()
        expansion = 4
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SingleConvNonAIELayerPost(nn.Module):
    def __init__(self, num_classes):
        super(SingleConvNonAIELayerPost, self).__init__()
        expansion = 4
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class NonAIELayerPost(nn.Module):
    def __init__(self, num_classes):
        super(NonAIELayerPost, self).__init__()
        expansion = 4
        self.linear = nn.Linear(64 * expansion, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PostConv2dLayers(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(PostConv2dLayers, self).__init__()

        self.in_planes = 256
        self.layer2 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Bottleneck_plain_without_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(Bottleneck_plain_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        out = self.conv3(out)
        out = out + self.shortcut(x)
        out = self.relu3(out)
        return out


class Bottleneck_plain_with_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(Bottleneck_plain_with_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = self.relu3(out)
        return out


class Bottleneck_projected_without_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(Bottleneck_projected_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, bias=False)
            )

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class triple_conv_without_bn(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(triple_conv_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        out = self.conv3(out)
        out = self.relu3(out)
        return out


class triple_conv_proper_without_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(triple_conv_proper_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        out = self.conv3(out)
        out = self.relu3(out)
        return out


class double_conv_withskip_without_bn(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(double_conv_withskip_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.shortcut = nn.Sequential()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        out = out + self.shortcut(x)
        out = self.relu3(out)
        return out


class triple_conv_withskip_without_bn(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(triple_conv_withskip_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class double_conv_without_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(double_conv_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(
            planes, 32, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        out = self.relu2((self.conv2(out)))
        return out


class single_conv_without_bn(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(single_conv_without_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1((self.conv1(x)))
        return out


class Bottleneck_base(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="B"):
        super(Bottleneck_base, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_projected(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(Bottleneck_projected, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = self.relu3(out)
        return out


# class AIELayerOffload(nn.Module):
#     def __init__(self, block, num_blocks):
#         super(AIELayerOffload, self).__init__()
#         self.in_planes = 64
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         print("strides:::",strides)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)


#     def forward(self, x):
#         # print("before",x.size())
#         out = self.layer1(x)
#         return out
class AIELayerOffloadPlain(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffloadPlain, self).__init__()
        self.in_planes = 256
        self.layer1 = block(in_planes=256, planes=64)
        # self.layer2 = block(in_planes=256,planes=64)
        # self.layer3 = block(in_planes=256,planes=64)

    def forward(self, x):
        # print("before",x.size())
        out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # print("after",out.size())
        return out


class AIELayerOffload(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffload, self).__init__()
        self.in_planes = 64
        self.layer1 = block(in_planes=64, planes=64)
        # self.layer2 = block(in_planes=256,planes=64)
        # self.layer3 = block(in_planes=256,planes=64)

    def forward(self, x):
        # print("before",x.size())
        out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # print("after",out.size())
        return out


class AIELayerOffload2(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffload2, self).__init__()
        self.in_planes = 64
        self.layer1 = block(in_planes=64, planes=64)
        self.layer2 = block(in_planes=256, planes=64)
        # self.layer3 = block(in_planes=256,planes=64)

    def forward(self, x):
        # print("before",x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # print("after",out.size())
        return out


class AIELayerOffload3(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffload3, self).__init__()
        self.in_planes = 64
        self.layer1 = block(in_planes=64, planes=64)
        self.layer2 = block(in_planes=256, planes=64)
        self.layer3 = block(in_planes=256, planes=64)

    def forward(self, x):
        # print("before",x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print("after",out.size())
        return out


class AIELayerOffload32(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffload32, self).__init__()
        self.in_planes = 32
        self.layer1 = block(in_planes=32, planes=32)
        # self.layer2 = block(in_planes=256,planes=64)
        # self.layer3 = block(in_planes=256,planes=64)

    def forward(self, x):
        # print("before",x.size())
        out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # print("after",out.size())
        return out


class NonOffloadBottleneck(nn.Module):
    def __init__(self, block, num_blocks):
        super(NonOffloadBottleneck, self).__init__()
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        return out


class CombinedModel(nn.Module):
    def __init__(self, first, aie, post):
        super(CombinedModel, self).__init__()
        self.first = first
        # self.pre = pre
        self.aie = aie
        self.post = post

    def forward(self, x):
        x = self.first(x)
        # x = self.pre(x)
        x = self.aie(x)
        x = self.post(x)
        return x


class CombinedModel_MIX(nn.Module):
    def __init__(self, first, pre, aie, post):
        super(CombinedModel_MIX, self).__init__()
        self.first = first
        self.pre = pre
        self.aie = aie
        self.post = post

    def forward(self, x):
        x = self.first(x)
        x = self.pre(x)
        x = self.aie(x)
        x = self.post(x)
        return x


class QuantSingleBottleneck(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(QuantSingleBottleneck, self).__init__()
        self.in_planes = 64
        self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(
            3,
            64,
            kernel_size=3,
            bias_quant=None,
            return_quant_tensor=False,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=8,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = qnn.QuantLinear(
            64 * block.expansion,
            num_classes,
            weight_bit_width=8,
            bias_quant=None,
            return_quant_tensor=False,
            bias=False,
        )
        self.relu1 = qnn.QuantReLU(bit_width=8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_inp(x)
        out = self.conv1(x)
        out = self.bn1(out)
        # print(out)
        out = self.relu1(out)
        out = self.layer1(out)
        out = F.avg_pool2d(out, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SingleBottleneck(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(SingleBottleneck, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.bn1(out)
        # print(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = F.avg_pool2d(out, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_projected(num_classes):
    return ResNet_projected(Bottleneck_projected, [2, 2, 2, 2], num_classes)


def ResNet34_projected(num_classes):
    return ResNet_projected(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50_projected(num_classes):
    return ResNet_projected(Bottleneck_projected, [3, 4, 6, 3], num_classes)


def ResNet101_projected(num_classes):
    return ResNet_projected(Bottleneck_projected, [3, 4, 23, 3], num_classes)


def ResNet152_projected(num_classes):
    return ResNet_projected(Bottleneck_projected, [3, 8, 36, 3], num_classes)


def SingleBottleneckModel_projected(num_classes):
    return SingleBottleneck(
        Bottleneck_projected,
        [
            1,
        ],
        num_classes,
    )


def QuantSingleBottleneckModel2x_projected(num_classes):
    return QuantSingleBottleneck(
        QuantBottleneck_projected,
        [
            2,
        ],
        num_classes,
    )


# def SingleBottleneckModel2x_projected_replicate(num_classes):
#     return SingleBottleneck(Bottleneck_projected, [2,],num_classes)


def SingleBottleneckModel2x_projected(num_classes):
    return SingleBottleneck(
        Bottleneck_projected,
        [
            2,
        ],
        num_classes,
    )


def DoubleBottleneckModel_projected(num_classes):
    return DoubleBottleneck(
        Bottleneck_projected,
        [
            2,
            2,
        ],
        num_classes,
    )


def TripleBottleneckModel_projected(num_classes):
    return TripleBottleneck(Bottleneck_projected, [1, 1, 1], num_classes)


def QuantFirstLayer():
    # very first resnet layer
    return QuantNonAIELayerInitial()


def QuantBottleneck_conv_Layer():
    # bottleneck with 1x1 conv in skip path
    return NonOffloadBottleneck(
        QuantBottleneck_projected,
        [
            1,
        ],
    )


def QuantAIELayer():
    # bottleneck with identiy in skip path
    return AIELayerOffload(
        QuantBottleneck_projected_OFFLOAD,
        [
            1,
        ],
    )


def QuantAIELayerFUSED():
    # bottleneck with identiy in skip path
    return AIELayerOffload(
        QuantBottleneck_projected_OFFLOAD_FUSED,
        [
            1,
        ],
    )


def QuantFinalLayer(num_classes):
    # final resnet layers
    return QuantNonAIELayerPost(num_classes)


def FirstLayer():
    # very first resnet layer
    return NonAIELayerInitial()


def Bottleneck_conv_Layer():
    # bottleneck with 1x1 conv in skip path
    return NonOffloadBottleneck(
        Bottleneck_projected,
        [
            1,
        ],
    )


def AIELayer():
    # bottleneck with identiy in skip path
    return AIELayerOffload(
        Bottleneck_projected_OFFLOAD,
        [
            1,
        ],
    )


def FinalLayer(num_classes):
    # final resnet layers
    return NonAIELayerPost(num_classes)


class Bottleneck_projected_OFFLOAD_BREAK(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, option="A", padding_mode="replicate"
    ):
        super(Bottleneck_projected_OFFLOAD_BREAK, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if padding_mode == "replicate":
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                padding_mode=padding_mode,
                stride=stride,
                padding=1,
                bias=False,
            )
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.conv4 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = self.relu3(out)
        out = self.relu4(self.bn4(self.conv4(x)))
        return out


class NonAIELayerPostBreak(nn.Module):
    def __init__(self, num_classes):
        super(NonAIELayerPostBreak, self).__init__()
        expansion = 4
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# IMAGENET MODELS


def ResNet50_projected_IMAGENET(num_classes):
    return ResNet_projected_IMAGENET(Bottleneck_projected, [3, 4, 6, 3], num_classes)


# __________________________________________________________________________________________
#  DEFINE FOR PTQ
#  Define AIE+HOST model

# def ResNet50_AIEConv2d_projected_NOQUANT(num_classes):
#     return CombinedModel(NonAIELayerInitial(),NonOffloadBottleneck(Bottleneck_projected, [1,]), \
#                          AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
#                               PostConv2dLayers(Bottleneck_projected, [4, 6, 3],num_classes))

# def Complete_SingleBottleneckModel2x_projected(num_classes):
#     return CombinedModel(NonAIELayerInitial(),NonOffloadBottleneck(Bottleneck_projected, [1,]), \
#                          AIELayerOffload(QuantBottleneck_projected_OFFLOAD, [1,]),\
#                               NonAIELayerPost(num_classes))

# def Complete_SingleBottleneckModel2x_projected_NOQUANT(num_classes):
#     return CombinedModel(NonAIELayerInitial(),NonOffloadBottleneck(Bottleneck_projected, [1,]), \
#                          AIELayerOffload(Bottleneck_projected_OFFLOAD, [1,]),\
#                               NonAIELayerPost(num_classes))


def Complete_SingleBottleneckModel3x_projected_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            Bottleneck_projected,
            [
                3,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


def Complete_SingleBottleneckModel1x_projected_NOQUANT_without_bn(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            Bottleneck_projected_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


def Complete_DoubleConvModel1x_projected_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            double_conv_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        DualConvNonAIELayerPost(num_classes),
    )


def Complete_DoubleConvModel_with_add1x_projected_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            double_conv_withskip_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        SingleConvNonAIELayerPost(num_classes),
    )


def Complete_TripleConvModel1x_projected_with_skip_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            triple_conv_withskip_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        SingleConvNonAIELayerPost(num_classes),
    )


def Complete_TripleConvModel1x_projected_without_skip_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            triple_conv_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        SingleConvNonAIELayerPost(num_classes),
    )


def Complete_SingleBottleneckModel1x_projected_NOQUANT_with_bn(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            Bottleneck_projected,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


def Complete_TripleConvModel1x_proper_projected_without_skip_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            triple_conv_proper_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


def Resnet50_2(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            triple_conv_proper_without_bn,
            [
                2,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        PostConv2dLayers(Bottleneck_projected, [2, 2, 2], num_classes),
    )


def Complete_TripleConvModel2x_proper_projected_without_skip_NOQUANT(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload2(
            triple_conv_proper_without_bn,
            [
                2,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


def Resnet50(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            triple_conv_proper_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        PostConv2dLayers(Bottleneck_projected, [2, 2, 2], num_classes),
    )


def SingleBottleneck_x1_plain(num_classes):
    return CombinedModel_MIX(
        NonAIELayerInitial(),
        NonOffloadBottleneck(
            Bottleneck_projected,
            [
                1,
            ],
        ),
        AIELayerOffloadPlain(
            Bottleneck_plain_without_bn,
            [
                1,
            ],
        ),
        NonAIELayerPost(num_classes),
    )


def SingleBottleneck_x1_plain_with_bn(num_classes):
    return CombinedModel_MIX(
        NonAIELayerInitial(),
        NonOffloadBottleneck(
            Bottleneck_projected,
            [
                1,
            ],
        ),
        AIELayerOffloadPlain(
            Bottleneck_plain_with_bn,
            [
                1,
            ],
        ),
        NonAIELayerPost(num_classes),
    )


def SingleBottleneck_x1_projected_with_bn(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload(
            Bottleneck_projected,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


# 2024.1.16
def SingleBottleneck_x2_projected_with_bn(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload2(
            Bottleneck_projected,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


def SingleBottleneck_x3_projected_with_bn(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload3(
            Bottleneck_projected,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )


# _________________________________________________________
# __________________________________________________________________________________________
# _________________________________________________________
# __________________________________________________________________________________________
# # 2024.1.25 oneRoof demo
# def Resnet50(num_classes):
#     return CombinedModel(NonAIELayerInitial(),AIELayerOffloadProper(Bottleneck_projected, [3,]), \
#                         #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
#                               PostConv2dLayers(Bottleneck_projected, [4,6,3],num_classes))
 
# # # 2024.1.26 oneRoof demo
# def Resnet50(num_classes):
#     return CombinedModel(NonAIELayerInitial(),AIELayerOffload3(Bottleneck_projected, [1,]), \
#                         #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
#                               PostConv2dLayers(Bottleneck_projected, [4,6,3],num_classes))
 
 
                            
# # # old ACDC demo
# def Resnet50ACDC(num_classes):
#     return CombinedModel(NonAIELayerInitial(),AIELayerOffload3(Bottleneck_projected, [1,]), \
#                         #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
#                               PostConv2dLayers(Bottleneck_projected, [2,2,2],num_classes))

def SingleBottleneck_x1_projected_without_bn(num_classes):
    return CombinedModel(NonAIELayerInitial(),AIELayerOffload(Bottleneck_projected_without_bn, [1,]), \
                        #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
                              NonAIELayerPost(num_classes))

def SingleBottleneck_x3_projected_without_bn(num_classes):
    return CombinedModel(
        NonAIELayerInitial(),
        AIELayerOffload3(
            Bottleneck_projected_without_bn,
            [
                1,
            ],
        ),  #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
        NonAIELayerPost(num_classes),
    )

# #  oneroof demo
def Resnet50ACDC(num_classes):
    return CombinedModel(NonAIELayerInitial(),AIELayerOffload3(Bottleneck_projected, [1,]), \
                        #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
                              PostConv2dLayers(Bottleneck_projected, [2,2,2],num_classes))
 
 

# #  oneroof demo
def Resnet50OneRoof(num_classes):
    return CombinedModel(NonAIELayerInitial(),AIELayerOffload3(Bottleneck_projected_without_bn, [1,]), \
                        #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
                              PostConv2dLayers(Bottleneck_projected, [2,2,2],num_classes))
 
 
# #  oneroof demo
def Resnet50(num_classes):
    return CombinedModel(NonAIELayerInitial(),AIELayerOffload3(Bottleneck_projected_without_bn, [1,]), \
                        #  AIELayerOffload(Bottleneck_projected_OFFLOAD, [2,]),\
                              PostConv2dLayers(Bottleneck_projected, [4,6,3],num_classes))
 