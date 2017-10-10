from torch.utils.data import Dataset

class KittyDataset(Dataset):
  """Kitty dataset"""
  DATA_DIR = '/home/ankitanand/PhDWork/Stereo_CNN_CRF/Datasets/Kitty/data_scene_flow/'
  def __init__(self, root, split='training', transform=False):
      self.root = root
      self.split = split
      self._transform = transform
      self.files = collections.defaultdict(list)
      for split in ['training', 'testing']:
            file_list = os.listdir(root + '/' + split+'/image_2')
            self.files[split] = file_list    
  

  def __len__(self):
    return len(self.files[self.split])


  def __getitem__(self, i):
    """
      Get the ith item from the dataset
      Return : left_img, right_img, target
    """
    img_name = self.files[self.split][index]
    left_img_path = self.root + '/' + self.split + '/myimage_2/' + img_name
    right_img_path= self.root + '/' + self.split + '/myimage_3/' + img_name 
    lbl_path = self.root + '/' + self.split + '/disp_noc_0/' + img_name

    left_img = m.imread(left_img_path)
    left_img = np.array(left_img, dtype=np.uint8)

    right_img =m.imread(right_img_path)
    right_img =np.array(right_img,dtype=np.uint8)

    lbl = m.imread(lbl_path)
    lbl = np.array(lbl, dtype=np.int32)

    if self.is_transform:
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