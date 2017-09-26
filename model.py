import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# TODO(AA) : Image reading
# TODO(AA) : CUDA compatible? / Can we implement correlation in C++/GPU (faster)?
#            Custom C++ implementation
# TODO(SG) : Correlation and rest of the module

# Global variables
k = 128

class Unary(nn.Module):
  """Unary CNN to get features from image"""
  def __init__(self, i):
    super(Unary, self).__init__()
    self.conv1 = nn.Conv2d(3, 100, 3, padding=1)
    self.layers = nn.ModuleList([nn.Conv2d(100, 100, 2) for _ in range(i-1)])
    self.padding = nn.ZeroPad2d((0, 1, 0, 1))
    print self.layers

  def forward(self, x):
    x = F.tanh(self.conv1(x))
    for layer in self.layers:
      x = F.tanh(layer(self.padding(x)))
    return x


class Correlation(torch.autograd.Function):
  def forward(self, left, right):
    """ Receive Tensor input, return output tensor"""
    # left, right are a 100 x w x h Tensor
    # for pos(i,j) we are to calculate left(:,i,j) * right(:,i+k,j)
    pass

  def backward(self, grad_output):
    """Calculate the gradients"""
    pass


def main():
  model = Unary(7)
  input = Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
  output = model(input)
  print output.size()

if __name__ == "__main__":
  main()