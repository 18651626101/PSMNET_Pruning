import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Function


class BinarizeFunc(Function):
  
  @staticmethod
  def forward(self,input):
    output = input.new(input.size())
    output[input>=0] = 1
    output[input<0] = -1
    return output

  @staticmethod
  def backward(self,grad_output):
    grad_input = grad_output.clone()
    return grad_input

# aliases
#binarize = BinarizeFunc.apply
def binarize(input):
    return BinarizeFunc()(input)


# quantization 
class QuantizeFunc(Function):
    
  @staticmethod
  def forward(self,input):
    b = 8   #bits
    output = input.new(input.size())
    output = (1.0/(2**b-1.0))*torch.round(input*(2**b-1))
    return output

  @staticmethod
  def backward(self,grad_output):
    grad_input  = grad_output.clone()
    return grad_input

def quantize(input):
  return QuantizeFunc()(input)

