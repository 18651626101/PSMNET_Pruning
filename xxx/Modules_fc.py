import math
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
from .Functions_quan import *

torch.manual_seed(42)



class QLinear(nn.Linear):

    def forward(self, input):
        qw = torch.tanh(self.weight)
        qw = qw/torch.max(torch.abs(qw)).data.item()*0.5+0.5
        qw = 2*QuantizeFunc.apply(qw)-1
        
        qb = torch.tanh(self.bias)
        qb = qb/torch.max(torch.abs(qb)).data.item()*0.5+0.5
        qb = 2*QuantizeFunc.apply(qb)-1

        return F.linear(input, qw, qb)
