import opengm
import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CRF(autograd.Function):
  """
    Takes the unary and pairwise potentials, do CRF inference over the given graphical model and return the labels
    Args:
      
      unary: unary potentials (b x k x r x c)
      pairwise: Format To be decided
    Return:
      Labels - (b x r x c)
  """
  def __init__(self, ):
    super(CRF, self).__init__()
    



  def forward(self, unary, pairwise):
    """ Receive input tensor, return output tensor"""
    self.save_for_backward(unary, pairwise)
    b,k,r,c = unary.size()
    unaries = unary.numpy()

    unaries = unaries.reshape([b*r*c,k])
    numVar = r*c
    
    gm=opengm.gm(numpy.ones(numVar,dtype=opengm.label_type)*k)
    uf_id = gm.addFunctions(unaries)
    potts = opengm.PottsFunction([k,k],0.0,0.4)
    pf_id = gm.addFunction(potts)

    vis=numpy.arange(0,numVar,dtype=numpy.uint64)
	# add all unary factors at once
	gm.addFactors(fids,vis)
	# add pairwise factors 
	### Row Factors
    for i in range(0,r):
		for j in range(0,c-1):
			gm.addFactor(pids,[i*c+j,i*c+j+1])
	### Column Factors
	for i in range(0,r-1):
		for j in range(c):
			gm.addFactor(pids,[i*c+j,(i+1)*c+j])
	print("Graphical Model Constructed")
	inf=opengm.inference.AlphaExpansionFusion(gm)
	inf.infer()
	labels=inf.arg()


		
 
   
    
    return torch.from_numpy(labels)
    

  def backward(self, grad_output):
    """Calculate the gradients of left and right"""
    print("Entering Backward Pass Through CRF\n Max Grad Outputs",torch.max(grad_output),torch.min(grad_output))
    unary, pairwise = self.saved_tensors
    
    print("Leaving Backward through CRF\n Min Grad Inputs",torch.max(l_grad),torch.min(l_grad))
    return l_grad, r_grad

width=100
height=200
numVar=width*height
numLabels=2
# construct gm
gm=opengm.gm(numpy.ones(numVar,dtype=opengm.label_type)*numLabels)
# construct an array with all numeries (random in this example)
unaries=numpy.random.rand(width,height,numLabels)
# reshape unaries is such way, that the first axis is for the different functions
unaries2d=unaries.reshape([numVar,numLabels])
# add all unary functions at once (#numVar unaries)
fids=gm.addFunctions(unaries2d)
# numpy array with the variable indices for all factors
vis=numpy.arange(0,numVar,dtype=numpy.uint64)
# add all unary factors at once
gm.addFactors(fids,vis)
print("Graphical Model Constructed")