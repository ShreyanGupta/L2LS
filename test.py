from __future__ import print_function
from torch.autograd import Variable
import torch

xx = Variable(torch.randn(1,1), requires_grad = True)
yy = 3*xx
zz = yy**2
print("x",xx)
print("y",yy)
yy.register_hook(print)
zz.backward()

