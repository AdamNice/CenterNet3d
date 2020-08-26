import time

import numpy as np
import spconv
import torch
from torch import nn
from torchplus.tools import change_default_args
from mmdet3d.models.registry import MIDDLE_ENCODERS
from ..activations import Mish
from .middle_aux import single_conv,double_conv,stride_conv,triple_conv


BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)



@MIDDLE_ENCODERS.register_module()
class SpMiddleFHD(nn.Module):
    def __init__(self,sparse_shape,in_channels=128,out_channels=128,activation="relu",name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name


        print("input sparse shape is ", sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = sparse_shape

        self.activation_fcn=change_default_args(inplace=True)(nn.ReLU)
        if activation=="lrelu":
            self.activation_fcn=change_default_args(negative_slope=0.1,inplace=True)(nn.LeakyReLU)
        if activation=="mish":
            self.activation_fcn=Mish

        # input: # [1600, 1200, 40]
        self.conv0 = double_conv(in_channels, 16, 'subm0', activation=self.activation_fcn)
        self.down0 = stride_conv(16, 32, 'down0', activation=self.activation_fcn)

        self.conv1 = double_conv(32, 32, 'subm1', activation=self.activation_fcn)  # [20,800,704]
        self.down1 = stride_conv(32, 64, 'down1', activation=self.activation_fcn)

        self.conv2 = triple_conv(64, 64, 'subm2', activation=self.activation_fcn)  # [10,400,352]
        self.down2 = stride_conv(64, 64, 'down2', activation=self.activation_fcn)

        self.conv3 = triple_conv(64, 64, 'subm3', activation=self.activation_fcn)  # [5,200,176]

        self.down3=spconv.SparseSequential(
            SpConv3d(64, 64, (3,1,1), (2,1,1), indice_key="down3"),
            BatchNorm1d(64),
            self.activation_fcn())                                           # [2,200,176]


    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        x= spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        x=self.conv0(x)
        x=self.down0(x)

        x=self.conv1(x)
        x=self.down1(x)

        x=self.conv2(x)
        x=self.down2(x)

        x=self.conv3(x)
        x=self.down3(x)

        ret = x.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

