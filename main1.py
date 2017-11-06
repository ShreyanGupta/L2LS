from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import MiddleburyDataset
from dataset import KittyDataset
from stereocnn import StereoCNN
from compute_error import compute_error

parser = argparse.ArgumentParser(description='StereoCNN model')
parser.add_argument('-k', "--disparity", type=int, default=256)
parser.add_argument('-ul', "--unary-layers", type=int, default=7)
parser.add_argument('-data', "--dataset", type=str, default="Middlebury")

parser.add_argument('-lr', "--learning-rate", type=float, default=1e-2)
parser.add_argument('-m', "--momentum", type=float, default=0.1)
parser.add_argument('-b', "--batch-size", type=int, default=1)
parser.add_argument('-n', "--num-epoch", type=int, default=100)
parser.add_argument('-v', "--verbose", type=bool, default=True)
args = parser.parse_args()

# Global variables
k = args.disparity
unary_layers = args.unary_layers
dataset=args.dataset
learning_rate = args.learning_rate
momentum = args.momentum
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = 1
verbose=args.verbose

def print_grad(grad):
  print("Grad_Max")
  print(torch.max(grad))

# DATA_DIR = '/Users/ankitanand/Desktop/Stereo_CRF_CNN/Datasets/Kitty/data_scene_flow'
#DATA_DIR = '/Users/Shreyan/Downloads/Datasets/Kitty/data_scene_flow'
DATA_DIR = '/home/ankit/Stereo_CNN_CRF/Datasets/'
save_path = "saved_model/model.pkl"

def main():
  if(dataset=="Middlebury"):
	train_set = MiddleburyDataset(DATA_DIR)
  else:
	train_set = KittyDataset(DATA_DIR)
  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  
  model = StereoCNN(unary_layers, k)
  loss_fn = nn.CrossEntropyLoss(size_average=False,ignore_index=-1)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  if torch.cuda.is_available():
	model = model.cuda()

  for epoch in range(num_epoch):
	print("epoch", epoch)
	for i, data in enumerate(train_loader):
	  left_img, right_img, labels = data
	  # No clamping might be dangerous
	  labels=labels.clamp_(-1,k)

	  optimizer.zero_grad()
	  if torch.cuda.is_available():
		left_img = left_img.cuda()
		right_img = right_img.cuda()
		labels=labels.cuda()
	 
	  left_img = Variable(left_img,requires_grad=True)
	  right_img = Variable(right_img,requires_grad=True)
	  labels = Variable(labels.type('torch.cuda.LongTensor'))
	  left,right= model(left_img,right_img)
	  b,d,r,c = left.size()
	  corr=Variable(torch.zeros(b,k,r,c).cuda())

	 
	  print("Left",left.size())
	  
	  for i in range(k):
		corr[:,i,:,0:c-k] = (left[:,:,:,0:c-k]*right[:,:,:,i:c-k+i]).sum(1)
	  for i in range(k):	
	  	for j in range(k-i):	 
	  		corr[:,j,:,c-k+i]=(left[:,:,:,c-k+i]*right[:,:,:,c-k+i+j]).sum(1)

	  # for i in range(c-k):
	  # 	for j in range(k):
	  # 		if(i+j<c):
	  # 			corr1[:,j,:,i]=(left[:,:,:,i]*right[:,:,:,i+j]).sum(1)
	  # print("Norm_Diff",corr-corr1)		
				
	  print(corr.size())
	  # pad = Variable(torch.cuda.FloatTensor(b,d,r,k).zero_())
	  # right = torch.cat([right, pad], dim=3)
	  # corr_vec = [(left*right.narrow(3,i,c)).sum(1) for i in range(k)]
	  # print("Corr_Vect",len(corr_vec[0].size()))
	  # y_pred = torch.stack(corr_vec, dim=1)
	  #y_pred = model(left_img, right_img)
	  y_pred_perm = corr.permute(0,2,3,1)
	  y_pred_perm.register_hook(print)
	  y_pred_ctgs = y_pred_perm.contiguous()
	  y_pred_flat= y_pred_ctgs.view(-1,k)
	  y_pred_.register_hook(print)
	  y_vals, y_labels = torch.max(y_pred_ctgs, dim=3)
	 
	  loss = loss_fn(y_pred_flat, labels.view(-1))
	 
	  
	  loss.backward()
	 
	  optimizer.step()
	  
	  error = compute_error(i, y_labels.data.cpu().numpy(), labels.data.cpu().numpy())
	  # error = 0
	  if(verbose):
		print("loss, error", i, loss.data[0], error)
	torch.save(model, save_path)

if __name__ == "__main__":
  main()
