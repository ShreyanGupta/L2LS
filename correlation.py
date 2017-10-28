import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

class Correlation(autograd.Function):
  def __init__(self, k):
    super(Correlation, self).__init__()
    self.k = k

  def forward(self, left, right):
    """ Receive Tensor input, return output tensor"""
    # left, right are a batch x 100 x w x h Tensor
    # return a batch x L x w x h Tensor (L is the number of labels)
    self.save_for_backward(left, right)
    left = left.numpy()
    right = right.numpy()
    b, _, w, h = left.shape
    output = np.empty((b, 0, w, h))
    for i in range(self.k):
      print "i =", i
      zero = np.zeros((b, i, h))
      l = left[:, :, 0:w - i, :]
      r = right[:, :, i:w, :]
      layer = np.einsum('abcd,abcd->acd', l, r)
      layer = np.concatenate((layer, zero), axis=1)
      output = np.concatenate((output, layer.reshape(b, 1, w, h)), axis=1)
    output_tensor = torch.from_numpy(output)
    return output_tensor

  def backward(self, grad_output):
    """Calculate the gradients"""
    # grad_output c is batch x k x w x h tensor del(L)/del(c)
    print("in Backward")
    left, right = self.saved_tensors
    b, d, w, h = left.shape
    l_grad = np.zeros((b, d, w, h))
    r_grad = np.zeros((b, d, w, h))
    for i in range(self.k):
      zero = np.zeros((b, d, i, h))
      temp_l_grad = np.multiply(grad_output[:, i:i + 1, i:w, h], left[:, :, 0:w - i, h])
      temp_r_grad = np.multiply(grad_output[:, i:i + 1, 0:w - i, h], right[:, :, i:w, h])
      l_grad += np.concatenate((zero, temp_l_grad), axis=2)
      r_grad += np.concatenate((temp_r_grad, zero), axis=2)
    l_grad = Variable(torch.from_numpy(l_grad.astype('float32')))
    r_grad = Variable(torch.from_numpy(r_grad.astype('float32')))
    print("end Backward")
    return l_grad, r_grad

k = 3
b, d, r, c = 1, 1, 2, 2

def another_correlation(left, right):
  print left
  right = torch.cat((right, Variable(torch.zeros(b,d,r,k))), dim=3)
  k_vec = torch.stack([(left*right.narrow(3,i,c)).sum(1) for i in range(k)], dim=1)
  return k_vec

  # print left
  # left = left.transpose(2,3)
  # left.contiguous()
  # left = left.view(d,-1)
  # left = torch.t(left)
  # left.contiguous()
  # right = right.transpose(2,3)
  # right.contiguous()
  # right = right.view(d,-1)
  # zero = torch.zeros(d, r*k)
  # right = torch.cat((right, zero), dim=1)
  # for i in range(k):
  #   print "i=",i, right[:, i*r:(i+c)*r]
  # k_vec = [torch.matmul(left, right[:, i*r:(i+c)*r]) for i in range(k)]
  # print k_vec[0]


if __name__ == "__main__":
  # For testing purpose
  arr = np.array([i for i in range(b*d*r*c)]).reshape((b,d,r,c))
  left = Variable(torch.FloatTensor(arr), requires_grad=True)
  right = Variable(torch.FloatTensor(arr), requires_grad=True)
  # output = Correlation(k)(l,r)
  ans = another_correlation(left, right)
  print ans
  ans = ans.sum()
  ans.backward()
  print ans
  print left.grad, right.grad
  # print(output)