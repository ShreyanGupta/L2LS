from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

from dataset import MiddleburyDataset
from dataset import KittyDataset
from stereocnn import StereoCNN
from compute_error import compute_error

parser = argparse.ArgumentParser(description='StereoCNN model')
parser.add_argument('-k', "--disparity", type=int, default=256)
parser.add_argument('-ul', "--unary-layers", type=int, default=3)
parser.add_argument('-data', "--dataset", type=str, default="Kitty")

parser.add_argument('-lr', "--learning-rate", type=float, default=1e-2)
parser.add_argument('-m', "--momentum", type=float, default=0.9)
parser.add_argument('-wd', "--weight-decay", type=float, default=0.9)
parser.add_argument('-b', "--batch-size", type=int, default=1)
parser.add_argument('-n', "--num-epoch", type=int, default=100)

parser.add_argument('-ms', "--model-file", type=str, default="model.pkl")
parser.add_argument('-ls', "--log-file", type=str, default="logs.txt")
args = parser.parse_args()

# Global variables
k = args.disparity
unary_layers = args.unary_layers
learning_rate = args.learning_rate
momentum = args.momentum
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = 4

# DATA_DIR = '/Users/ankitanand/Desktop/Stereo_CRF_CNN/Datasets/'
DATA_DIR = '/Users/Shreyan/Downloads/Datasets/'
# DATA_DIR = '/home/ankit/Stereo_CNN_CRF/Datasets/'

model_save_path = os.path.join("experiments", args.model_file)
log_file = open(os.path.join("experiments", args.log_file), "w")

def main():
  # Get dataset
  if(args.dataset=="Middlebury"):
    train_set = MiddleburyDataset(DATA_DIR)
  else:
    train_set = KittyDataset(DATA_DIR)
  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  
  # Model, loss, optimizer
  model = StereoCNN(unary_layers, k)
  loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)

  # Weight initialization
  for p in list(model.parameters()):
    if len(p.size()) >= 2:
      nn.init.xavier_normal(p)

  if torch.cuda.is_available():
    model = model.cuda()

  for l in list(model.parameters()):
    l.register_hook(lambda x: print("model grad", x.min().data[0], x.max().data[0]))

  for epoch in range(num_epoch):
    print("epoch", epoch)
    torch.save(model, model_save_path)
    scheduler.step()
    for i, data in enumerate(train_loader):
      left_img, right_img, labels = data
      # No clamping might be dangerous
      labels.clamp_(-1,k-1)

      if torch.cuda.is_available():
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        labels = labels.cuda()
      
      left_img = Variable(left_img)
      right_img = Variable(right_img)
      labels = Variable(labels)

      optimizer.zero_grad()
      y_pred = model(left_img, right_img)
      y_pred = y_pred.permute(0,2,3,1)
      y_pred = y_pred.contiguous()
      
      temp, y_labels = torch.max(y_pred, dim=3)
      # print temp[:, 50:60, 50:60], y_labels[:, 50:60, 50:60]
      print("y_labels", y_labels.min().data[0], y_labels.max().data[0])

      print "backward"
      loss = loss_fn(y_pred.view(-1,k), labels.view(-1))
      loss.backward()
      error = compute_error(i, y_labels.data.cpu().numpy(), labels.data.cpu().numpy())
      # error = 0
      for l in list(model.parameters()):
        print("model", l.min().data[0], l.max().data[0])
      print("loss", loss.data[0], "error", error)
      optimizer.step()
      error = compute_error(epoch, i, log_file, loss.data[0], y_labels.data.cpu().numpy(), labels.data.cpu().numpy())

if __name__ == "__main__":
  main()
