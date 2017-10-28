import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import KittyDataset
from stereocnn import StereoCNN
from compute_error import compute_error

parser = argparse.ArgumentParser(description='StereoCNN model')
parser.add_argument('-k', "--disparity", type=int, default=256)
parser.add_argument('-ul', "--unary-layers", type=int, default=7)

parser.add_argument('-lr', "--learning-rate", type=float, default=1e-2)
parser.add_argument('-m', "--momentum", type=float, default=0.1)
parser.add_argument('-b', "--batch-size", type=int, default=1)
parser.add_argument('-n', "--num-epoch", type=int, default=100)
args = parser.parse_args()

# Global variables
k = args.disparity
unary_layers = args.unary_layers

learning_rate = args.learning_rate
momentum = args.momentum
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = 4

# DATA_DIR = '/Users/ankitanand/Desktop/Stereo_CRF_CNN/Datasets/Kitty/data_scene_flow'
DATA_DIR = '/Users/Shreyan/Downloads/Datasets/Kitty/data_scene_flow'
# DATA_DIR = '/home/ankit/Stereo_CNN_CRF/Datasets/Kitty/data_scene_flow'
save_path = "saved_model/model.pkl"

def main():
  train_set = KittyDataset(DATA_DIR)
  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  
  model = StereoCNN(unary_layers, k)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  if torch.cuda.is_available():
    model = model.cuda()

  for epoch in range(num_epoch):
    print("epoch", epoch)
    for i, data in enumerate(train_loader):
      left_img, right_img, labels = data
      # No clamping might be dangerous
      labels.clamp_(0,k-1)

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
      _, y_labels = torch.max(y_pred, dim=3)
      
      loss = loss_fn(y_pred.view(-1,k), labels.view(-1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      error = compute_error(i, y_labels.data.numpy(), labels.data.numpy())
      # error = 0
      print("loss, error", i, loss.data[0], error)
    torch.save(model, save_path)

if __name__ == "__main__":
  main()
