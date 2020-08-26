import torch.nn as nn
import torch

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[3, 5, 9],input_planes=160):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        self.block=nn.Sequential(nn.Conv2d(input_planes*4,input_planes,3,padding=1,bias=False),
                                 nn.BatchNorm2d(input_planes),
                                 nn.LeakyReLU(inplace=True))

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        features=self.block(features)
        return features