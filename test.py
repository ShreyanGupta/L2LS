import numpy as np

k = 2
# batxh x 100 x w x h
x = np.ones((1,2,5,5))
y = np.ones((1,2,5,5))

print x.shape, y.shape
b, _, w, h = x.shape
ans = np.zeros((b,0,w,h))

for i in xrange(k):
  p = x[:, :, 0:w-i, :]
  q = y[:, :, i:w, :]
  r = np.einsum('abcd,abcd->acd', p, q)
  zero = np.zeros((b,i,h))
  cat = np.concatenate((r,zero), axis=1)
  ans = np.concatenate((ans,cat.reshape((b,1,w,h))), axis=1)
  print p.shape, q.shape, r.shape, zero.shape, cat.shape, ans.shape