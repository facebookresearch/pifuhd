import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json 
import numpy as np
import cv2
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.visualizer import Visualizer
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index

opt = BaseOptions().parse()

def precompute_points(opt):
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    dataset = RPOtfDataset(opt, phase='all')

    subjects = dataset.get_subjects()
    
    print('# of subjects: ', len(subjects))
    dataset.opt.simga = 5
    dataset.opt.sampleing_mode = 'uniform_simga5'

    for sub in subjects:
        dataset.precompute_points(sub, num_files=200)

    # for i in range(1, 10, 2):
    #     dataset.opt.sigma = float(i)
    #     dataset.opt.sampleing_mode = 'uniform_simga%d' % i
    #     for sub in subjects:
    #         if i == 1 and sub == 'rp_andrew_posed_002':
    #             flag = True
    #         if not flag:
    #             continue
    #         print(sub, 'sigma', i)
    #         try:
    #             dataset.precompute_points(sub, num_files=10)
    #         except:
    #             print('failed', sub)

def precompute_tsdf(opt):
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    dataset = RPOtfDataset(opt, phase='train')

    subjects = dataset.get_subjects()

    for sub in subjects:
        dataset.precompute_tsdf(sub, num_files=100, sigma=3.0)


if __name__ == '__main__':
    precompute_points(opt)
    # precompute_tsdf(opt)
  