import numpy as np

error_px = 3
error_percent = 0.05

def compute_error(batch_num, prediction, gt):
  diff = np.abs(prediction - gt)
  #wrong_elems = (gt > 0) & (diff > error_px) & ((diff / np.abs(gt)) > error_percent)
  wrong_elems = (gt > 0) & (diff > error_px)
  error = wrong_elems.sum() / np.sum(gt[gt > 0].shape).astype('float64')
  fp = open('Logs.txt',"a")
  fp.write('batch_num: '+str(batch_num)+' error: '+str(error))
  fp.close()
  return error
