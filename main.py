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

# TODO(??) : Add a profiler

# Global variables
k = 255
learning_rate = 1e-2
momentum = 0.1


def main():
  # x is input, y is output
  train_set = KittyDataset()
  train_loader = DataLoader(train_set, batch_size=1, num_workers=4, shuffle=True)
  #val_set = voc.VOC('val', transform=input_transform, target_transform=target_transform)
  #val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)
  #loader = DataLoader(VOC12(args.datadir, input_transform, target_transform),
  #      num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
  #x = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
  #y = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
  
  model = StereoCNN(7, k)
  loss_fn = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for i in range(100):
    for i, data in enumerate(train_loader):
        left_img, right_img, labels = data
        left_img= Variable(left_img,requires_grad=True)
        right_img = Variable(right_img,requires_grad=True)
        labels=Variable(labels)


        labels=labels.view(-1)

        print(labels)
        #print(y_indices)
        y_pred = model(left_img, right_img)
        # vals,y_indices=torch.max(y_pred,1)
        y_pred = torch.transpose(y_pred, 1, 3)
        y_pred = y_pred.contiguous()
        y_pred = y_pred.view(-1, k)

        y_pred=y_pred.type(torch.FloatTensor)
        print(y_pred)
        labels=torch.clamp(labels, min=1, max=k-2)
        labels=labels.type(torch.LongTensor)
        loss = loss_fn(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i,loss.data)

#def test():
#  x = torch.autograd.Variable(torch.Tensor(np.zeros((1, 3, 250, 250))))
#  model = Unary(7)
#  y_pred = model(x)
    #print y_pred.size()

if __name__ == "__main__":
  main()
#test()
