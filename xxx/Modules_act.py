import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.Functions_quan import *

class BinaryTanh(nn.Module):

  def __init__(self):
    super(BinaryTanh, self).__init__()
    self.hardtanh = nn.Hardtanh()

  def forward(self, input):
  
    output = self.hardtanh(input)
    output = binarize(output)
    return output


class QuantizeActivation(nn.Module):

  def __init__(self):
    super(QuantizeActivation, self).__init__()
    self.hardtanh = nn.Hardtanh()

  def forward(self, input):
    output = (self.hardtanh(input)+1.0)/2
    output = QuantizeFunc.apply(output)
    return output


class PrunActivation(nn.Module):

  def __init__(self,threshold):
    super(PrunActivation, self).__init__()
    self.prun = nn.Threshold(threshold, 0)
    self.hardtanh = nn.Hardtanh()
    

  def forward(self, input):
    
    output = (self.hardtanh(input)+1.0)/2
    output = QuantizeFunc.apply(output)
    output = self.prun(output)
    return output



