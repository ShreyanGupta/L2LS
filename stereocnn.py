import torch.nn as nn
import torch
from torch.autograd import Variable

from unary import Unary
from correlation import Correlation,correlation

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
    #self.correlation=Correlation(self.k)


  def forward(self, l, r):
<<<<<<< Updated upstream
    phi_left = self.unary_left(l)
    phi_right = self.unary_right(r)
    return phi_left,phi_right
    #corr=self.correlation(phi_left,phi_right)
    #corr = Correlation(self.k)(phi_left, phi_right)
    #corr=correlation(phi_left,phi_right,self.k)
    #return corr
 
=======
    phi_left = self.unary(l)
    phi_right = self.unary(r)
   
    corr = Correlation(self.k)(phi_left, phi_right)
    return corr
>>>>>>> Stashed changes
