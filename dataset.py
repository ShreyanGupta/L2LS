from torch.utils.data import Dataset
import collections
import os
import scipy.misc as m
import numpy as np

class KittyDataset(Dataset):
  """Kitty dataset"""
  def __init__(self, root, split='training', transform=None):
    self.root = root
    self.split = split
    self._transform = transform
    self.files = collections.defaultdict(list)
    for split in ['training', 'testing']:
      file_list = os.listdir(root + '/' + split+'/myimage_2')
      self.files[split] = file_list

  def __len__(self):
    return len(self.files[self.split])

  def __getitem__(self, i):
    """
      Get the ith item from the dataset
      Return : left_img, right_img, target
    """
    img_name = self.files[self.split][i]
    left_img_path = self.root + '/' + self.split + '/myimage_2/' + img_name
    right_img_path= self.root + '/' + self.split + '/myimage_3/' + img_name 
    lbl_path = self.root + '/' + self.split + '/disp_noc_0/' + img_name

    left_img = m.imread(left_img_path)
    left_img = np.array(left_img, dtype=np.float32)
    left_img = left_img.transpose(2,0,1)
    left_img = (left_img - left_img.mean())/left_img.std()
    # print "left", left_img.mean(), left_img.std()

    right_img = m.imread(right_img_path)
    right_img = np.array(right_img,dtype=np.float32)
    right_img = right_img.transpose(2,0,1)
    right_img = (right_img - right_img.mean())/right_img.std()
    # print "right", right_img.mean(), right_img.std()


    # TODO(AA) : Figure out the right representation of imread
    # TODO(AA) : Normalize the input as per paper
    lbl = m.imread(lbl_path)
    lbl = np.array(lbl, dtype=np.int64)

    if self._transform:
      left_img, right_img, lbl = self.transform(img, lbl)

    return left_img, right_img, lbl
    

class MiddleburyDataset(Dataset):
  """Middlebury dataset"""
  def __init__(self, path):
    self.path = path

  def __len__(self):
    return -1

  def __getitem__(self, i):
    """
      Get the ith item from the dataset
      Return : left_img, right_img, target
    """
    pass