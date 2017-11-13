from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

from unary import Unary
from correlation import Correlation

class StereoCNN(nn.Module):
  """Stereo vision module"""
  def __init__(self, i, k, unary_path=None):
    """Args:
      i (int): Number of layers in the Unary units
      k (int): Disparity label count
      unary_path (string option): Path of unary model to load
    """
    super(StereoCNN, self).__init__()
    self.k = k
    if unary_path:
      print("loading unary from", unary_path)
      self.unary = torch.load(unary_path)
    else:
      print("init new unary")
      self.unary = Unary(i)
      # Weight initialization
      for p in list(self.parameters()):
        if len(p.size()) >= 2:
          nn.init.xavier_normal(p)

  def forward(self, l, r):
    phi_left = self.unary(l)
    phi_right = self.unary(r)
    corr = Correlation(self.k)(phi_left, phi_right)
    return corr

  def save_unary(self, model_save_path):
    torch.save(self.unary, model_save_path)
