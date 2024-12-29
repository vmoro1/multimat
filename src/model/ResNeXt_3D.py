# 3D ResNeXt architecture for encoding of charge density. Note: Downsampling block is rather upsampling block.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

# from utils import count_parameters  # TODO fix path

__all__ = ['ResNeXt', 'resnext50', 'resnext18','resnext101', 'resnext152']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()

    zero_pads = zero_pads.to(device)

    out = torch.cat([out.data, zero_pads], dim=1) 

    return out


class ResNeXtBottleneck(nn.Module):
    """By what factor the number of filters are reduced in the bottleneck (e.g 2 mean that you halve the number of dimensions). 
    If this is modified it would also make sense to modify the number of filters in _make_layer()"""
    expansion = 2   

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)    
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, 
                 block,
                 layers,
                 sample_depth=32,
                 sample_height=32,
                 sample_width=32,
                 shortcut_type='B',
                 cardinality=32,
                 in_channels=1,
                 embedding_dim=128,
                 projector=None,
                 batch_norm=False):
        """sample_depth, sample_height, sample_width should be the spatial dimensions of the input data: (N,C,D,H,W).
        Shortcut type should be either 'A' or 'B'."""
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        
        last_depth = int(math.ceil(sample_depth / 32)) 
        last_height = int(math.ceil(sample_height / 32))
        last_width = int(math.ceil(sample_width / 32))

        self.avgpool = nn.AvgPool3d(
            (last_depth, last_height, last_width), stride=1)
        # self.fc = nn.Linear(cardinality * 32 * block.expansion, embedding_dim)
        self.fc = nn.Linear(2048, embedding_dim)
        if projector:
            self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        else:
            self.proj = None
        if batch_norm:
            self.final_bn = nn.BatchNorm1d(embedding_dim, affine=False)
        else:
            self.final_bn = None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        """planes refer to the numer of filters in the middle block in the bottleneck"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:   # If shortcut type == 'B'
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.proj is not None:
            x = self.proj(x)
        if self.final_bn is not None:
            x = self.final_bn(x)

        return x

def resnext18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNeXt(ResNeXtBottleneck, [2, 2, 2, 2], **kwargs)
    return model

def resnext50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    model = resnext50(sample_depth=32, sample_height=32, sample_width=32, shortcut_type='B', cardinality=32, in_channels=3, embedding_dim=128)
    x = torch.rand(2,3,32,32,32)
    output = model(x)
    print(output.size())
    print(f"Number of parameters: {count_parameters(model)}")
