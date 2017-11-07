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

  # def forward(self, left, right):
  #   """ Receive input tensor, return output tensor"""
  #   self.save_for_backward(left, right)
  #   self.type = left.type()
  #   b,d,r,c = left.size()
  #   pad = torch.zeros(b,d,r,self.k).type(self.type)
  #   right = torch.cat((right, pad), dim=3)
  #   corr_vec = [(left * right[:, :, :, i:i+c]).sum(dim=1) for i in range(self.k)]
  #   return torch.stack(corr_vec, dim=1)

  def forward(self, left, right):
    """ Receive input tensor, return output tensor"""
    self.save_for_backward(left, right)
    self.type = left.type()
    b,d,r,c = left.size()
    pad = torch.zeros(b,d,r,self.k-1).type(self.type)
    right = torch.cat((pad, right), dim=3)
    corr_vec = [(left * right[:, :, :, self.k-1-i:self.k-1-i+c]).sum(dim=1) for i in range(self.k)]
    return torch.stack(corr_vec, dim=1)

  # def backward(self, grad_output):
  #   """Calculate the gradients of left and right"""
  #   left, right = self.saved_tensors
  #   b,d,r,c = left.size()
  #   pad = torch.zeros(b,d,r,self.k).type(self.type)
  #   right = torch.cat((right, pad), dim=3)
  #   left = torch.cat((pad, left), dim=3)
  #   l_grad = torch.zeros(b,d,r,c).type(self.type)
  #   r_grad = torch.zeros(b,d,r,c).type(self.type)
  #   for i in range(self.k):
  #     l_grad += grad_output[:, i:i+1, :, :] * right[:, :, :, i:i+c]
  #     r_grad += grad_output[:, i:i+1, :, :] * left[:, :, :, self.k-i:self.k-i+c]
  #   return l_grad, r_grad

  def backward(self, grad_output):
    """Calculate the gradients of left and right"""
    left, right = self.saved_tensors
    b,d,r,c = left.size()
    pad = torch.zeros(b,d,r,self.k-1).type(self.type)
    right = torch.cat((pad, right), dim=3)
    left = torch.cat((left, pad), dim=3)
    l_grad = torch.zeros(b,d,r,c).type(self.type)
    r_grad = torch.zeros(b,d,r,c).type(self.type)
    for i in range(self.k):
      l_grad += grad_output[:, i:i+1, :, :] * right[:, :, :, self.k-1-i:self.k-1-i+c]
      r_grad += grad_output[:, i:i+1, :, :] * left[:, :, :, i:i+c]
    return l_grad, r_grad



# Testing

# k = 2
# b, d, r, c = 1, 5, 4, 6

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