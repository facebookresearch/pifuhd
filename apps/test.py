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

from lib.options import BaseOptions
from lib.visualizer import Visualizer
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index

parser = BaseOptions()

def reshape_multiview_tensors(image_tensor, calib_tensor):
    '''
    args:
        image_tensor: [B, nV, C, H, W]
        calib_tensor: [B, nV, 3, 4]
    return:
        image_tensor: [B*nV, C, H, W]
        calib_tensor: [B*nV, 3, 4]
    '''
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor

def reshape_sample_tensor(sample_tensor, num_views):
    '''
    args:
        sample_tensor: [B, 3, N] xyz coordinates
        num_views: number of views
    return:
        [B*nV, 3, N] repeated xyz coordinates
    '''
    if num_views == 1:
        return sample_tensor
    sample_tensor = sample_tensor[:, None].repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor

def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=False):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=500000)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        net.calc_normal(verts_tensor, calib_tensor[:1])
        color = net.nmls.detach().cpu().numpy()[0].T
        # xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        # uv = xyz_tensor[:, :2, :]
        # color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)

def total_error(opt, errors, multi_gpu=False):
    if multi_gpu:
        for key in errors.keys():
            errors[key] = errors[key].mean()

    error = 0
    error += errors['Err(occ)']
    if 'Err(nml)' in errors:
        error += opt.lambda_nml * errors['Err(nml)']
    if 'Err(L1)' in errors:
        error += opt.lambda_cmp_l1 * errors['Err(L1)']
    return error

def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests, label=None, thresh=0.5):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_arr, IOU_arr, prec_arr, recall_arr = {}, [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)
            if opt.num_sample_normal:
                sample_nml_tensor = data['samples_nml'].to(device=cuda).unsqueeze(0)
                label_nml_tensor = data['labels_nml'].to(device=cuda).unsqueeze(0)
            else:
                sample_nml_tensor = None
                label_nml_tensor = None

            error, res = net(image_tensor, sample_tensor, calib_tensor, label_tensor,\
                             sample_nml_tensor, label_nml_tensor)

            err = total_error(opt, error, False)

            IOU, prec, recall = compute_acc(res, label_tensor, thresh)

            if idx == 0:
                error_arr['Err(total)'] = [err.item()]
                for k, v in error.items():
                    error_arr[k] = [v.item()]
            else:
                error_arr['Err(total)'].append(err.item())
                for k, v in error.items():
                    error_arr[k].append(v.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())
    
    if label is not None:
        err = {}
        for k, v in error_arr.items():
            err['%s-%s' % (k, label)] = np.average(v)
        acc = {
            'IOU-%s' % label: np.average(IOU_arr),
            'prec-%s' % label: np.average(prec_arr),
            'recall-%s' % label: np.average(recall_arr),
        }       
        err.update(acc)     
        return err
    else:
        err = {}
        for k, v in error_arr.items():
            err['%s' % k] = np.average(v)
        acc = {
            'IOU': np.average(IOU_arr),
            'prec': np.average(prec_arr),
            'recall': np.average(recall_arr),
        }       
        err.update(acc)     
        return err

def eval(opt):
    # load checkpoints
    state_dict_path = None
    if opt.load_netG_checkpoint_path is not None:
        state_dict_path = opt.load_netG_checkpoint_path
    elif opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    else:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    
    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path)    
        if 'opt' in state_dict:
            print('Warning: opt is overwritten.')
            opt = state_dict['opt']
    else:
        raise Exception('failed loading state dict!', state_dict_path)
    
    parser.print_options(opt)

    cuda = torch.device('cuda:%d' % opt.gpu_id)

    if opt.use_tsdf:
        test_dataset = RPTSDFDataset(opt, phase='val')
    elif opt.sampling_otf:
        test_dataset = RPOtfDataset(opt, phase='val')
    else:
        test_dataset = RPDataset(opt, phase='val')

    print('test data size: ', len(test_dataset))
    projection_mode = test_dataset.projection_mode

    ls_thresh = 0.5 # level set boundary
    criteria = {'occ': nn.MSELoss()}
    if opt.nml_loss_type == 'mse':
        criteria['nml'] = nn.MSELoss()
    elif opt.nml_loss_type == 'l1':
        criteria['nml'] = nn.L1Loss()

    netG = HGPIFuNet(opt, projection_mode, criteria).to(device=cuda)

    def set_eval():
        netG.eval()

    # load checkpoints
    if state_dict is not None:
        if 'model_state_dict' in state_dict:
            netG.load_state_dict(state_dict['model_state_dict'])
        else: # this is deprecated but keep it for now.
            netG.load_state_dict(state_dict)

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s/eval' % (opt.results_path, opt.name), exist_ok=True)

    ## test
    with torch.no_grad():
        set_eval()

        if not opt.no_numel_eval:
            print('calc error (test) ...')
            err = calc_error(opt, netG, cuda, test_dataset, 100, 'test')
            print('eval: ', ''.join(['{}: {:.6f} '.format(k, v) for k,v in err.items()]))

        if not opt.no_mesh_recon:
            print('generate mesh (test) ...')
            for test_data in tqdm(test_dataset):
                save_path = '%s/%s/eval/test_eval_%s_%d_%d.obj' % (
                    opt.results_path, opt.name, test_data['name'], test_data['vid'], test_data['pid'])
                gen_mesh(opt.resolution, netG, cuda, test_data, save_path)

def evalWrapper(args=None):
    opt = parser.parse(args)
    eval(opt)

if __name__ == '__main__':
    evalWrapper()
  