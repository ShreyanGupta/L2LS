import numpy as np

error_px = 3
error_percent = 0.05


def ComputeError(batch_num, prediction, gt ):
    diff = np.abs(prediction - gt)
    wrong_elems = (gt > 0) & (diff > error_px) & ((diff / np.abs(gt)) > error_percent)

    error = wrong_elems.sum() / np.sum(gt[gt > 0].shape).astype('float64')
    fp=open('Logs.txt',"a")
    fp.write('batch_num:',batch_num,'error:',error)
    fp.close()
    return error