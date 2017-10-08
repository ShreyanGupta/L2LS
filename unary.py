import torch.nn as nn
import torch.nn.functional as F

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