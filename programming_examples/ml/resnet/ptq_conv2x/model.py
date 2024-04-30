import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedModel(nn.Module):
    def __init__(self, first, aie, post):
        super(CombinedModel, self).__init__()
        self.first = first
        self.aie = aie
        self.post = post

    def forward(self, x):
        x = self.first(x)
        x = self.aie(x)
        x = self.post(x)
        return x

class PreAIELayers(nn.Module):
    def __init__(self):
        super(PreAIELayers, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # print( out)
        out = F.relu(out)
        return out


class AIEConv2xOffload(nn.Module):
    def __init__(self, block, num_blocks):
        super(AIEConv2xOffload, self).__init__()
        self.in_planes = 64
        self.layer1 = block(in_planes=64, planes=64)
        self.layer2 = block(in_planes=256, planes=64)
        self.layer3 = block(in_planes=256, planes=64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class PostAIELayers(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(PostAIELayers, self).__init__()

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

class Bottleneck_fused_projected(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(Bottleneck_fused_projected, self).__init__()
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
        
def Resnet50_conv2x_offload(num_classes):
    return CombinedModel(
        PreAIELayers(),
        AIEConv2xOffload(
            Bottleneck_fused_projected,
            [
                1,
            ],
        ), 
        PostAIELayers(Bottleneck_projected, [4, 6, 3], num_classes),
    )