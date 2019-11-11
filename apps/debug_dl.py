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
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.options import BaseOptions
from torchy.data import *


parser = BaseOptions()
        
def train(opt):
    parser.print_options(opt)

    dataset = FRLFaceDataset(opt, phase='train')
    print(len(dataset))
    # dataset = RPDatasetParts(opt, phase='train')
    # if opt.use_tsdf:
    #     dataset = RPTSDFDataset(opt, phase='train')
    # elif opt.sampling_otf:
    #     dataset = RPOtfDataset(opt, phase='train')
    # else:
    #     dataset = RPDataset(opt, phase='train')
    # dataset[0]
    data_loader = DataLoader(dataset,
                                   batch_size=opt.batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('data size: ', len(data_loader))

    for train_idx, data in tqdm(enumerate(data_loader)):
        pass

def trainerWrapper(args=None):
    opt = parser.parse(args)
    train(opt)

if __name__ == '__main__':
    trainerWrapper()