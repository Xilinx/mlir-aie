import torch
import torch.nn as nn
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np


# sys.path.append("../../misc");
# from utils import count_parameters
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


class AIELayerOffload2(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIELayerOffload2, self).__init__()
        self.in_planes = 64
        self.layer1 = block(in_planes=64, planes=64)
        self.layer2 = block(in_planes=256, planes=64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


input = torch.randn(1, 64, 32, 32)
offload_model = AIELayerOffload2(Bottleneck_projected, [2])
offload_model.eval()
fp_out = offload_model(input)
# print(fp_out)
