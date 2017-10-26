import torch
import torch.nn as nn
import torch.nn.functional as F

class Unary(nn.Module):
  """Unary CNN to get features from image"""
  def __init__(self, i):
    super(Unary, self).__init__()
    self.features=nn.Sequential(
      nn.Conv2d(3, 100, 3, padding=1),
      nn.Tanh(),
      # nn.Conv2d(100, 100, 2,padding=1),
      # nn.Tanh(),
      # nn.Conv2d(100, 100, 2,padding=0),
      # nn.Tanh(),
    )

    self.padding = nn.ZeroPad2d((0, 1, 0, 1))
    #print self.layers

  def forward(self, x):
    #print(x)
    x=torch.transpose(x,1,3)
    x_feat=self.features(x)
    return x_feat