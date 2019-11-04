from torch.utils.data import Dataset 
import random

class MixDataset(Dataset):
    '''
    This is base dataset class. never call it as is.
    The class provides the expected output from self.get_item
    '''
    def __init__(self, p_db, s_db, ratio=0.5, phase='train'):
        self.is_train = phase == 'train'

        self.primary = p_db
        self.secondary = s_db

        self.pdb_size = len(self.primary)
        self.sdb_size = len(self.secondary)

        self.ratio = ratio

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, index):
        if self.is_train:
            self.primary.is_train = True
            self.secondary.is_train = True
            if random.random() < self.ratio:
                return self.primary[index]
            else:
                ridx = random.randint(0, self.sdb_size-1)
                return self.secondary[ridx]
        else:
            self.primary.is_train = False
            self.secondary.is_train = False
            if index % 2 == 0:
                return self.primary[index]
            else:
                idx = index % self.sdb_size
                return self.secondary[idx]