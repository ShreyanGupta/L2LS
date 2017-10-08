import torch
import numpy as np

class Correlation(torch.autograd.Function):
  """Defines the correlation (till before the softmax layer)"""
  def forward(self, left, right):
    """ Receive Tensor input, return output tensor"""
    # left, right are a batch x 100 x w x h Tensor
    # return a batch x L x w x h Tensor (L is the number of labels)
    self.save_for_backward(left, right)
    left = left.numpy()
    right = right.numpy()
    b, _, w, h = left.shape
    output = np.empty((b,0,w,h))
    for i in xrange(k):
      zero = np.zeros((b,i,h))
      l = left[:, :, 0:w-i, :]
      r = right[:, :, i:w, :]
      layer = np.einsum('abcd,abcd->acd', l, r)
      layer = np.concatenate((layer, zero), axis=1)
      output = np.concatenate((output, layer.reshape(b,1,w,h)), axis=1)
    return output

  def backward(self, grad_output):
    """Calculate the gradients"""
    # grad_output c is batch x k x w x h tensor del(L)/del(c)
    left, right = self.saved_tensors
    left = left.numpy()
    right = right.numpy()
    b, d, w, h = left.shape
    l_grad = np.zeros((b,d,w,h))
    r_grad = np.zeros((b,d,w,h))
    for i in xrange(k):
      # TODO(SG) : Implement this Function
      pass