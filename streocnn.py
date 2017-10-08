import torch.nn as nn

from unary import Unary
from correlation import Correlation

class StereoCNN(nn.Module):
  """Stereo vision module"""
  def __init__(self, i):
    super(StereoCNN, self).__init__()
    # TODO(SG) : Zero mean and unit variance karna hai
    self.unary_left = Unary(i)
    self.unary_right = Unary(i)
    # TODO(SG) : check if this softmax is the required softmax
    self.softmax = nn.softmax2d()

  def forward(self, l, r):
    phi_left = self.unary_left(l)
    phi_right = self.unary_right(r)
    corr = Correlation.apply(phi_left, phi_right)
    # ?????????
    corr = self.softmax(corr)
    return corr