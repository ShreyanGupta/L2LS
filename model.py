import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(AA) : Image reading
# TODO(AA) : CUDA compatible? / Can we implement correlation in C++/GPU (faster)?
#            Custom C++ implementation
# TODO(SG) : Correlation and rest of the module

# Global variables
k = 128
learning_rate = 1e-2
momentum = 0.1

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
  """Defines the correlation (till before the softmax layer)"""
  # TODO(SG) : Implement this Function
  def forward(self, left, right):
    """ Receive Tensor input, return output tensor"""
    # left, right are a 100 x w x h Tensor
    # for pos(i,j) we are to calculate left(:,i,j) * right(:,i+k,j)
    # finally return a L x w x h Tensor (L is the number of labels)
    pass

  def backward(self, grad_output):
    """Calculate the gradients"""
    pass


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
    corr = self.softmax(corr)
    return corr


def main():
  # x is input, y is output
  x = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
  y = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
  
  model = StereoCNN(7)
  loss_fn = nn.L1Loss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for i in range(100):
    # TODO(AA) : fetch input and output
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print i, loss.data

def test():
  x = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
  model = Unary(7)
  y_pred = model(x)
  print y_pred.size()

if __name__ == "__main__":
  # main()
  test()