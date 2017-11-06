import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

class Correlation(autograd.Function):
  """
    Calculated the correlation between the left and the right images
    Args:
      k : number of labels
      left : left image of dim (b x d x r x c)
      right : right image of dim (b x d x r x c)
    Return:
      correlation of dim (b x k x r x c)
  """
  def __init__(self, k):
    super(Correlation, self).__init__()
    self.k = k

  def forward(self, left, right):
    """ Receive input tensor, return output tensor"""
    self.save_for_backward(left, right)
    self.type = left.type()
    b,d,r,c = left.size()
    pad = torch.zeros(b,d,r,self.k).type(self.type)
    right = torch.cat((right, pad), dim=3)
    corr_vec = [(left * right[:, :, :, i:i+c]).sum(dim=1) for i in range(self.k)]
    return torch.stack(corr_vec, dim=1)

  def backward(self, grad_output):
    """Calculate the gradients of left and right"""
    left, right = self.saved_tensors
    b,d,r,c = left.size()
    pad = torch.zeros(b,d,r,self.k).type(self.type)
    right = torch.cat((right, pad), dim=3)
    left = torch.cat((pad, left), dim=3)
    l_grad = torch.zeros(b,d,r,c).type(self.type)
    r_grad = torch.zeros(b,d,r,c).type(self.type)
    for i in range(self.k):
      l_grad += grad_output[:, i:i+1, :, :] * right[:, :, :, i:i+c]
      r_grad += grad_output[:, i:i+1, :, :] * left[:, :, :, self.k-i:self.k-i+c]
    return l_grad, r_grad


# Old correlation

# class Correlation(autograd.Function):
#   def __init__(self, k):
#     super(Correlation, self).__init__()
#     self.k = k

#   def forward(self, left, right):
#     """ Receive Tensor input, return output tensor"""
#     # left, right are a batch x 100 x w x h Tensor
#     # return a batch x L x w x h Tensor (L is the number of labels)
#     self.save_for_backward(left, right)
#     left = left.numpy()
#     right = right.numpy()
#     b, _, w, h = left.shape
#     output = np.empty((b, 0, w, h))
#     for i in range(self.k):
#       print "i =", i
#       zero = np.zeros((b, i, h))
#       l = left[:, :, 0:w - i, :]
#       r = right[:, :, i:w, :]
#       layer = np.einsum('abcd,abcd->acd', l, r)
#       layer = np.concatenate((layer, zero), axis=1)
#       output = np.concatenate((output, layer.reshape(b, 1, w, h)), axis=1)
#     output_tensor = torch.from_numpy(output)
#     return output_tensor

  # def backward(self, grad_output):
  #   """Calculate the gradients"""
  #   # grad_output c is batch x k x w x h tensor del(L)/del(c)
  #   print("in Backward")
  #   left, right = self.saved_tensors
  #   b, d, w, h = left.shape
  #   l_grad = np.zeros((b, d, w, h))
  #   r_grad = np.zeros((b, d, w, h))
  #   for i in range(self.k):
  #     zero = np.zeros((b, d, i, h))
  #     temp_l_grad = np.multiply(grad_output[:, i:i + 1, i:w, h], left[:, :, 0:w - i, h])
  #     temp_r_grad = np.multiply(grad_output[:, i:i + 1, 0:w - i, h], right[:, :, i:w, h])
  #     l_grad += np.concatenate((zero, temp_l_grad), axis=2)
  #     r_grad += np.concatenate((temp_r_grad, zero), axis=2)
  #   l_grad = Variable(torch.from_numpy(l_grad.astype('float32')))
  #   r_grad = Variable(torch.from_numpy(r_grad.astype('float32')))
  #   print("end Backward")
  #   return l_grad, r_grad


# Testing

# k = 2
b, d, r, c = 1, 5, 4, 6

def correlation(left, right, k):
  b,d,r,c = left.size()
  pad = Variable(torch.zeros(b,d,r,k))
  right = torch.cat((right, pad), dim=3)
  corr_vec = [(left*right.narrow(3,i,c)).sum(1) for i in range(k)]
  return torch.stack(corr_vec, dim=1)

if __name__ == "__main__":
  arr = np.array([i for i in range(b*d*r*c)]).reshape((b,d,r,c))
  
  left = Variable(torch.FloatTensor(arr), requires_grad=True)
  right = Variable(torch.FloatTensor(arr), requires_grad=True)
  left2 = Variable(torch.FloatTensor(arr), requires_grad=True)
  right2 = Variable(torch.FloatTensor(arr), requires_grad=True)
  
  ans = Correlation(k)(left,right)
  ans2 = another_correlation(left2, right2)
  # print "correlation", ans, ans2
  ans = ans.sum()
  ans2 = ans2.sum()
  ans.backward()
  ans2.backward()
  print ans
  print "left grads", left.grad.data.equal(left2.grad.data)
  print "right grads", right.grad.data.equal(right2.grad.data)
  # print(output)