import numpy as np
import torch
import torch.nn as nn

from stereocnn import StereoCNN

# TODO(AA) : Image reading
# TODO(AA) : CUDA compatible? / Can we implement correlation in C++/GPU (faster)?
#            Custom C++ implementation
# TODO(SG) : Correlation and rest of the module
# TODO(??) : Add a profiler

# Global variables
k = 128
learning_rate = 1e-2
momentum = 0.1


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