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

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck_CIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Bottleneck_CIFAR, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

       

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SingleBottleneckAIE(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(SingleBottleneckAIE, self).__init__()
        self.in_planes = 64

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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




class NonAIELayerInitial(nn.Module):
    def __init__(self):
        super(NonAIELayerInitial, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    def forward(self, x):
        out=self.conv1(x)
        out = self.bn1(out)
        # print( out)
        out = F.relu(out)
        return out
    
class NonAIELayerPost(nn.Module):
    def __init__(self,  num_classes):
        super(NonAIELayerPost, self).__init__()
        expansion = 4
        self.linear = nn.Linear(64*expansion, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class AIELayerOffload(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffload, self).__init__()
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        return out

class SingleBottleneck_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(SingleBottleneck_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.linear = nn.Linear(16, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
def ResNet18_CIFAR(num_classes):
    return ResNet_CIFAR(Bottleneck_CIFAR, [2, 2, 2, 2],num_classes)


def SingleBottleneckModel_CIFAR(num_classes):
    return SingleBottleneck_CIFAR(Bottleneck_CIFAR, [1,],num_classes)


# def ResNet50(num_classes):
#     return ResNet_CIFAR(Bottleneck, [3, 4, 6, 3],num_classes)


# def ResNet101(num_classes):
#     return ResNet_CIFAR(Bottleneck, [3, 4, 23, 3],num_classes)


# def ResNet152(num_classes):
#     return ResNet_CIFAR(Bottleneck, [3, 8, 36, 3],num_classes)



# def SingleBottleneckModel2x(num_classes):
#     return SingleBottleneck(Bottleneck, [2,],num_classes)

# def DoubleBottleneckModel(num_classes):
#     return DoubleBottleneck(Bottleneck, [2, 2, ],num_classes)

# def TripleBottleneckModel(num_classes):
#     return TripleBottleneck(Bottleneck, [1, 1, 1],num_classes)

# def FirstLayer():
#     return NonAIELayerInitial()

# def AIELayer():
#     return AIELayerOffload(Bottleneck, [1,])

def FinalLayer(num_classes):
    return NonAIELayerPost(num_classes)