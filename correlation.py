import torch
import numpy as np

class Correlation(torch.autograd.Function):
  """Defines the correlation (till before the softmax layer)"""
  def forward(self, left, right, k):
    """ Receive Tensor input, return output tensor"""
    # left, right are a batch x 100 x w x h Tensor
    # return a batch x L x w x h Tensor (L is the number of labels)
    self.save_for_backward(left, right)
    self.k = k
    left = left.numpy()
    right = right.numpy()
    b, _, w, h = left.shape
    output = np.empty((b,0,w,h))
    for i in xrange(self.k):
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
    for i in xrange(self.k):
      zero = np.zeros((b,d,i,h))
      temp_l_grad = np.multiply(grad_output[:, i:i+1, i:w, h], left[:, :, 0:w-i, h])
      temp_r_grad = np.multiply(grad_output[:, i:i+1, 0:w-i, h], right[:, :, i:w, h])
      l_grad += np.concatenate((zero, temp_l_grad), axis=2)
      r_grad += np.concatenate((temp_r_grad, zero), axis=2)
    return l_grad, r_grad


if __name__ == "__main__":
  # For testing purpose
  # TODO(AA) : Please test this function
  k = 5
  b, w, h = 10, 50, 50
  l = torch.FloatTensor(np.array([i for i in range(b*25*w*h)]).reshape((b,25,w,h)))
  r = torch.FloatTensor(np.array([i for i in range(b*25*w*h)]).reshape((b,25,w,h)))
  output = Correlation.apply(l,r,k)
  print output