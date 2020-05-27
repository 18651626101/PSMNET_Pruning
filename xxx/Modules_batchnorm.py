import math
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch

from .Functions_quan import *

torch.manual_seed(42)


class QBatchNorm2d(nn.BatchNorm2d):

  def forward(self, input):
    qw = torch.tanh(self.weight)
    qw = qw/torch.max(torch.abs(qw)).data.item()*0.5+0.5
    qw = 2*QuantizeFunc.apply(qw)-1
    qb = torch.tanh(self.bias)
    qb = qb/torch.max(torch.abs(qb)).data.item()*0.5+0.5
    qb = 2*QuantizeFunc.apply(qb)-1

    return F.batch_norm(
            input, self.running_mean, self.running_var, qw, qb,
            self.training or not self.track_running_stats,
            self.momentum, self.eps)

class QBatchNorm3d(nn.BatchNorm3d):

  def forward(self, input):
    qw = torch.tanh(self.weight)
    qw = qw/torch.max(torch.abs(qw)).data.item()*0.5+0.5
    qw = 2*QuantizeFunc.apply(qw)-1
    qb = torch.tanh(self.bias)
    qb = qb/torch.max(torch.abs(qb)).data.item()*0.5+0.5
    qb = 2*QuantizeFunc.apply(qb)-1

    return F.batch_norm(
            input, self.running_mean, self.running_var, qw, qb,
            self.training or not self.track_running_stats,
            self.momentum, self.eps)