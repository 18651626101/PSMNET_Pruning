import math
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch

from .Functions_quan import *

torch.manual_seed(42)



class QConv2d(nn.Conv2d):
    def forward(self, input):
        qw = torch.tanh(self.weight)
        qw = qw/torch.max(torch.abs(qw)).data.item()*0.5+0.5
        qw = 2*QuantizeFunc.apply(qw)-1
        if self.bias is not None:
            qb = torch.tanh(self.bias)
            qb = qb/torch.max(torch.abs(qb)).data.item()*0.5+0.5
            qb = 2*QuantizeFunc.apply(qb)-1
            return F.conv2d(input, qw, qb, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, qw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class QConv3d(nn.Conv3d):
    def forward(self, input):
        qw = torch.tanh(self.weight)
        qw = qw/torch.max(torch.abs(qw)).data.item()*0.5+0.5
        qw = 2*QuantizeFunc.apply(qw)-1
        if self.bias is not None:
            qb = torch.tanh(self.bias)
            qb = qb/torch.max(torch.abs(qb)).data.item()*0.5+0.5
            qb = 2*QuantizeFunc.apply(qb)-1
            return F.conv3d(input, qw, qb, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, qw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
