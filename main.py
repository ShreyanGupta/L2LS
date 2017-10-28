import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import KittyDataset
from stereocnn import StereoCNN

# TODO(AA) : CUDA compatible? / Can we implement correlation in C++/GPU (faster)?
#            Custom C++ implementation
# TODO(??) : Add a profiler
# TOGO(??) : Complete the CRF wala part

# Global variables
k = 256
learning_rate = 1e-3
momentum = 0.1

batch_size = 1
num_workers = 4
# DATA_DIR = '/Users/ankitanand/Desktop/Stereo_CRF_CNN/Datasets/Kitty/data_scene_flow'
# DATA_DIR = '/Users/Shreyan/Downloads/Datasets/Kitty/data_scene_flow'
DATA_DIR = '/home/ankit/Stereo_CNN_CRF/L2LS/Datasets/Kitty/data_scene_flow'

def main():
  train_set = KittyDataset(DATA_DIR)
  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  
  model = StereoCNN(1, k)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  if torch.cuda.is_available():
    model = model.cuda()

  for epoch in range(100):
    print("epoch", epoch)
    for i, data in enumerate(train_loader):
      left_img, right_img, labels = data
      # No clamping might be dangerous
      # labels.clamp_(0,k-1)

      if torch.cuda.is_available():
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        labels = labels.cuda()
      
      left_img = Variable(left_img)
      right_img = Variable(right_img)
      labels = Variable(labels)

      y_pred = model(left_img, right_img)
      y_pred = y_pred.permute(0,2,3,1)
      y_pred = y_pred.contiguous()
      
      loss = loss_fn(y_pred.view(-1,k), labels.view(-1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print("loss", i, loss.data[0])
      break

if __name__ == "__main__":
  main()
