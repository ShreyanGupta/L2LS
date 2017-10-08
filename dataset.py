from torch.utils.data import Dataset

class KittyDataset(Dataset):
  """Kitty dataset"""
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