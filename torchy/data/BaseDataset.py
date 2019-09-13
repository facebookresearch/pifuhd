from torch.utils.data import Dataset 
import random

class BaseDataset(Dataset):
    '''
    This is base dataset class. never call it as is.
    The class provides the expected output from self.get_item
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', projection='orthogonal'):
        self.opt = opt
        self.is_train = phase == 'train'
        self.projection_mode = projection

    def __len__(self):
        return 0

    def get_item(self, index):
        # in case of IO error, use random sampling instead
        subject = ''
        try:
            res = {
                'name': None, # name of the subject
                'b_min': None, # bbox (x_min, y_min, z_min) of the target space
                'b_max': None, # bbox (x_max, y_max, z_max) of the target space

                'samples': None, # [3, N] 3d points
                'labels': None, # [1, N] labels

                'img': None, # [num_views, C, H, W] input images
                'calib': None, # [num_views, 4, 4] calibration matrix
                'extrinsic': None, # [num_views, 4, 4] extrinsic matrix
                'mask': None, # [num_views, 1, H, W] segmentation masks
            }
            return res
        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)