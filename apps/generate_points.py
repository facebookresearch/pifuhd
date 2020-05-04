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
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index
from tqdm import tqdm

parser = BaseOptions()

def precompute_points(opt):
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    dataset = RPOtfDataset(opt, phase='all')

    subjects = dataset.get_subjects()
    
    print('# of subjects: ', len(subjects))
    for sub in tqdm(subjects):
        dataset.precompute_points(sub, num_files=2, start_id=2*opt.tmp_id, sigma=3.0) # you can change sigma

def pgWrapper(args=None):
    opt = parser.parse(args)
    precompute_points(opt)

# import submitit
def submit():
    base_cmd =['--dataroot', '/run/media/hjoo/disk/data/pifuhd/data/pifu_data', '--num_sample_inout', '500000', '--sampling_mode', 'sigma3_uniform']

#     executor = submitit.AutoExecutor(folder="tmp_cluster_log")  # submission interface (logs are dumped in the folder)
#     executor.update_parameters(timeout_min=3*60, gpus_per_node=1, cpus_per_task=10, partition="priority", name='wildPIFu', comment='cvpr deadline')  # timeout in min

    for i in range(0,50):
        cmd = base_cmd + ['--tmp_id', '%d' % i]
        pgWrapper(cmd)
        # job = executor.submit(pgWrapper, cmd)  
        # print(job.job_id)  # ID of your job

if __name__ == '__main__':
    submit()
  