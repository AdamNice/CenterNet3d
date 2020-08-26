from torch import nn
from ..activations import Mish

class DepthWiseBlock(nn.Module):
    def __init__(self, in_channel,out_channel,activation="relu"):
        super(DepthWiseBlock, self).__init__()

        activation_fun=nn.ReLU()
        if activation=="lrelu":
            activation_fun=nn.LeakyReLU(0.1,inplace=True)
        elif activation=="mish":
            activation_fun=Mish()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride=1,padding=1,dilation=1,groups=in_channel),
            ##深度卷积，因为输入的分组数g取in_channel，故输出通道数就是g个(即in_channel)
            nn.BatchNorm2d(in_channel,momentum=0.01),
            activation_fun,
            ##inplace=True表示进行原地操作，对上一层传递下来的tensor直接进行修改，如x=x+3。这样可以节省运算内存，不用多存储变量
            nn.Conv2d(in_channel, out_channel, 1, 1),
            # 点卷积，用1*1不会改大小，上步把图已经缩减为原来的1/4，为了不发生因卷积而信息丢失(参数量减少)，这步需要把通道数扩增为4倍
            nn.BatchNorm2d(out_channel,momentum=0.01),
            activation_fun,
        )

    def forward(self, input):
        output=self.sequential(input)
        return output
