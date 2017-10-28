import torch.nn as nn
import torch
from torch.autograd import Variable

from unary import Unary
# from correlation import Correlation

def correlation(left, right, k):
  b,d,r,c = left.size()
  pad = Variable(torch.zeros(b,d,r,k))
  right = torch.cat((right, pad), dim=3)
  corr_vec = [(left*right.narrow(3,i,c)).sum(1) for i in range(k)]
  return torch.stack(corr_vec, dim=1)

class StereoCNN(nn.Module):
  """Stereo vision module"""
  def __init__(self, i, k):
    """Args:
        i (int): Number of layers in the Unary units
        k (int): Disparity label count
    """
    super(StereoCNN, self).__init__()
    self.k = k
    self.unary_left = Unary(i)
    self.unary_right = Unary(i)

  def forward(self, l, r):
    print "begin unary"
    phi_left = self.unary_left(l)
    phi_right = self.unary_right(r)
    print "begin corr"
    # corr = Correlation(self.k)(phi_left, phi_right)
    corr = correlation(phi_left, phi_right, self.k)
    print corr.size()
    print "exit forward"
    return corr