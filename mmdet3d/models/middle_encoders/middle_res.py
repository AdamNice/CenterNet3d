import time

import numpy as np
import spconv
import torch
from torch import nn
from torchplus.tools import change_default_args
from ..registry import MIDDLE_ENCODERS
from ..activations import Mish

BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)

def SparseConv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)


def SparseConv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)



class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, activation_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = SparseConv3x3(inplanes, planes, stride, indice_key=indice_key)
        self.bn1 = BatchNorm1d(planes)
        self.relu = activation_fn()
        self.conv2 = SparseConv3x3(planes, planes, indice_key=indice_key)
        self.bn2 = BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


@MIDDLE_ENCODERS.register_module()
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, sparse_shape,in_channels=128,out_channels=128,activation="relu",name='SpMiddleResFHD'
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.sparse_shape = sparse_shape
        self.voxel_output_shape = sparse_shape

        activation_fun = nn.ReLU
        if activation == "lrelu":
            activation_fun = nn.LeakyReLU
        elif activation == "mish":
            activation_fun = Mish

        # input: # [1600, 1200, 40]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(in_channels, 16, 3, indice_key="res0"),
            BatchNorm1d(16),
            activation_fun(0.1,inplace=True),
            # SparseBasicBlock(16, 16, indice_key="res0",activation_fn=activation_fun),
            SparseBasicBlock(16, 16, indice_key="res0",activation_fn=activation_fun),
            SpConv3d(
                16, 32, 3, 2, padding=1,
            ),  # [1600, 1200, 40] -> [800, 600, 20]
            BatchNorm1d(32,),
            activation_fun(),
            # SparseBasicBlock(32, 32, indice_key="res1",activation_fn=activation_fun),
            SparseBasicBlock(32, 32, indice_key="res1",activation_fn=activation_fun),
            SpConv3d(
                32, 64, 3, 2, padding=1,
            ),  # [800, 600, 20] -> [400, 300, 10]
            BatchNorm1d(64),
            activation_fun(),
            SparseBasicBlock(64, 64,indice_key="res2",activation_fn=activation_fun),
            SparseBasicBlock(64, 64,indice_key="res2",activation_fn=activation_fun),
            SpConv3d(
                64, 64, 3, 2, padding=[1, 1, 1],
            ),  # [400, 300, 10] -> [200, 150, 5]
            BatchNorm1d(64,),
            activation_fun(),
            SparseBasicBlock(64,64, indice_key="res3",activation_fn=activation_fun),
            SparseBasicBlock(64,64, indice_key="res3",activation_fn=activation_fun),
            SpConv3d(
                64,64, (3, 1, 1), (2, 1, 1)
            ),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            activation_fun()
        )

    def forward(self, voxel_features, coors, batch_size):


        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret
