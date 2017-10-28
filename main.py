import numpy as np
import torch
import torch.nn as nn

from dataset import KittyDataset
import scipy.misc as m
from torch.utils.data import DataLoader
from torch.autograd import Variable
from stereocnn import StereoCNN

# TODO(AA) : CUDA compatible? / Can we implement correlation in C++/GPU (faster)?
#            Custom C++ implementation
# TODO(AA) : Correlation Testing
# TODO(SG) : Complete the StereoCNN
# TODO(??) : Add a profiler
# TOGO(??) : Complete the CRF wala part

# Global variables
k = 10
learning_rate = 1e-2
momentum = 0.1
batch_size = 1
num_workers = 4
# DATA_DIR = '/Users/ankitanand/Desktop/Stereo_CRF_CNN/Datasets/Kitty/data_scene_flow'
DATA_DIR = '/Users/Shreyan/Downloads/Datasets/Kitty/data_scene_flow'

def main():
  # x is input, y is output
  train_set = KittyDataset(DATA_DIR)
  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  
  model = StereoCNN(1, k)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for i in range(100):
    for i, data in enumerate(train_loader):
      left_img, right_img, labels = data
      
      # TODO(SG) : Remove the clamp once label is resolved
      labels.clamp_(0,k-1)
      print left_img.size()
      
      left_img = Variable(left_img)
      right_img = Variable(right_img)
      labels = Variable(labels)

      y_pred = model(left_img, right_img)
      y_pred = y_pred.permute(0,2,3,1)
      y_pred = y_pred.contiguous()
      
      print "y_pred", y_pred.size()
      loss = loss_fn(y_pred.view(-1,k), labels.view(-1))
      print "done loss", loss
      optimizer.zero_grad()
      print "start backward"
      loss.backward()
      print "end backward, start optimizer"
      optimizer.step()
      "print end optimizer"
      print(i,loss.data)

#def test():
#  x = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
#  model = Unary(7)
#  y_pred = model(x)
    #print y_pred.size()

if __name__ == "__main__":
  main()
#test()
