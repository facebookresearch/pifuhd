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
from lib.visualizer import Visualizer
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index
from torchy.net_util import CustomBCELoss, CustomMSELoss

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

def gen_mesh(res, netC, cuda, data, save_path, thresh=0.5, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    netC.filter(image_tensor, with_netG=True)

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
            netC.netG, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        color = np.zeros(verts.shape)
        interval = 100000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print('Can not create marching cubes at this time.', e)

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def total_error(opt, errors, multi_gpu=False):
    # NOTE: in multi-GPU case, since forward returns errC with number of GPUs, we need to marge.
    if multi_gpu:
        for key in errors.keys():
            errors[key] = errors[key].mean()

    error = 0
    error += errors['Err(clr)']

    return error

def calc_error(opt, netC, cuda, dataset, num_tests, label=None, thresh=0.5):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_arr = {}
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            errC, res = netC(image_tensor, sample_tensor, calib_tensor, label_tensor)

            err = total_error(opt, errC, False)

            if idx == 0:
                error_arr['Err(total)'] = [err.item()]
                for k, v in errC.items():
                    error_arr[k] = [v.item()]
            else:
                error_arr['Err(total)'].append(err.item())
                for k, v in errC.items():
                    error_arr[k].append(v.item())
    
    if label is not None:
        err = {}
        for k, v in error_arr.items():
            err['%s-%s' % (k, label)] = np.average(v)
        return err
    else:
        err = {}
        for k, v in error_arr.items():
            err['%s' % k] = np.average(v)
        return err
        
def train(opt):
    # load checkpoints
    state_dict_path = None
    if opt.load_netC_checkpoint_path is not None:
        state_dict_path = opt.load_netC_checkpoint_path
        opt.continue_train = True
    elif opt.continue_train and opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    elif opt.continue_train:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    else:
        opt.continue_train = False
        opt.resume_epoch = 0
    
    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path)    
        if not opt.finetune and 'opt' in state_dict:
            print('Warning: opt is overwritten.')
            continue_train = opt.continue_train
            opt = state_dict['opt']
            opt.continue_train = continue_train
    elif state_dict_path is not None:
        print('Error: unable to load checkpoint %s' % state_dict_path)

    writer = SummaryWriter(log_dir="./logs/%s" % opt.name)            
    
    parser.print_options(opt)

    cuda = torch.device('cuda:%d' % opt.gpu_id)

    if opt.sampling_parts:
        train_dataset = RPDatasetParts(opt, phase='train')
        test_dataset = RPDatasetParts(opt, phase='val')
    else:
        train_dataset = RPDataset(opt, phase='train')
        test_dataset = RPDataset(opt, phase='val')

    projection_mode = train_dataset.projection_mode

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_data_loader))

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    ls_thresh = 0.5 # level set boundary
    criteria = {}
    if opt.clr_loss_type == 'mse':
        criteria['clr'] = nn.MSELoss()
    elif opt.clr_loss_type == 'l1':
        criteria['clr'] = nn.L1Loss()
    else:
        raise NameError('unknown loss type %s' % opt.clr_loss_type)
        
    if opt.load_netG_checkpoint_path is not None:
        print('Loading netG from ', opt.load_netG_checkpoint_path)
        state_dict_netG = torch.load(opt.load_netG_checkpoint_path) 
        state_dict_netG['opt'].merge_layer = opt.merge_layer
        if state_dict_netG['opt'].netG == 'hghpifu':
            netG = HGHPIFuNet(state_dict_netG['opt'], projection_mode)
        else:
            netG = HGPIFuNet(state_dict_netG['opt'], projection_mode)

        if state_dict_netG is not None:
            if 'model_state_dict' in state_dict_netG:
                netG.load_state_dict(state_dict_netG['model_state_dict'])
            else: # this is deprecated but keep it for now.
                netG.load_state_dict(state_dict_netG)
    else:
        print('Warning: netG is initialized!!')

    if opt.netC == 'resblkhpifu':
        netC = ResBlkHPIFuNet(opt, netG, projection_mode, criteria)
    else:
        netC = ResBlkPIFuNet(opt, netG, projection_mode, criteria)
    lr = opt.learning_rate
    
    def set_train():
        netC.train()

    def set_eval():
        netC.eval()

    # load checkpoints        
    if state_dict is not None:
        if 'model_state_dict' in state_dict:
            netC.load_state_dict(state_dict['model_state_dict'])
        else: # this is deprecated but keep it for now.
            netC.load_state_dict(state_dict)
    
        if 'epoch' in state_dict:
            opt.resume_epoch = state_dict['epoch']
        if opt.resume_epoch < 0 or opt.finetune:
            opt.resume_epoch = 0
 
    netC = netC.to(device=cuda)
    
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netC = nn.DataParallel(netC)
        multi_gpu = True

    optimizerC = torch.optim.Adam(netC.parameters(), lr=opt.learning_rate)

    if not opt.finetune and state_dict is not None and 'optimizer_state_dict' in state_dict:
        optimizerC.load_state_dict(state_dict['optimizer_state_dict'])

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # training
    start_epoch = 0 if opt.finetune or not opt.continue_train else opt.resume_epoch
    num_epoch = 1 + opt.num_iter // len(train_data_loader)
    cur_iter = state_dict['cur_iter'] if not opt.finetune and state_dict is not None and 'cur_iter' in state_dict \
                else start_epoch * len(train_data_loader)

    for epoch in range(start_epoch, num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['color_samples'].to(device=cuda)
            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

            label_tensor = train_data['rgbs'].to(device=cuda)

            iter_start_time = time.time()

            errC, res = netC(image_tensor, sample_tensor, calib_tensor, label_tensor)
            err = total_error(opt, errC, multi_gpu)

            optimizerC.zero_grad()
            err.backward()
            optimizerC.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_save_ply == 0 and train_idx != 0:
                save_path = '%s/%s/test_epoch%d_idx%d.ply' % (opt.results_path, opt.name, epoch, train_idx)
                r = 0.5*res[0].transpose(0,1).cpu().detach().numpy()+0.5
                points = sample_tensor[0].transpose(0, 1).cpu().detach().numpy()
                save_samples_rgb(save_path, points, r)
  
            if train_idx % opt.freq_plot == 0:
                print(
                    'Name: %s | Epoch: %d/%d | %d/%d | Err: %.5f | %sLR: %.1e | SIG: %.1e | dataT: %.4f | netT: %.4f | ETA: %02d:%02d' % (
                        opt.name, epoch, num_epoch, train_idx, len(train_data_loader), err.item(), ''.join(['{}: {:.4f} | '.format(k, v.item()) for k,v in errC.items()]),
                        lr, opt.sigma, iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),
                        int(eta - 60 * (eta // 60))))
                counter_ratio = train_idx / len(train_data_loader)
                losses = {}
                losses['Err(total)'] = err.item()
                for k, v in errC.items():
                    losses[k] = v.item()
                for k, v in losses.items():
                    writer.add_scalar('%s/train-runtime' % k, v, cur_iter)

            if cur_iter % opt.freq_eval == 0 and cur_iter != 0:
                with torch.no_grad():
                    test_losses = {}
                    if not opt.no_numel_eval:
                        set_eval()
                        print('calc error (train) ...')
                        train_dataset.is_train = False
                        err = calc_error(opt, netC if not multi_gpu else netC.module, cuda, train_dataset, 50, 'train', ls_thresh)
                        train_dataset.is_train = True
                        print('eval: ', ''.join(['{}: {:.6f} '.format(k, v) for k,v in err.items()]))
                        test_losses.update(err)
                        for k, v in err.items():
                            writer.add_scalar('%s/%s' % (k.split('-')[0],'train'), v, cur_iter)

                        print('calc error (test) ...')
                        err = calc_error(opt, netC if not multi_gpu else netC.module, cuda, test_dataset, 100, 'test', ls_thresh)
                        print('eval: ', ''.join(['{}: {:.6f} '.format(k, v) for k,v in err.items()]))
                        test_losses.update(err)
                        for k, v in err.items():
                            writer.add_scalar('%s/%s' % (k.split('-')[0],'test'), v, cur_iter)

                        set_train()

                    save_dict = {
                        'opt': opt,
                        'epoch': epoch,
                        'cur_iter': cur_iter,
                        'model_state_dict': netC.state_dict() if not multi_gpu else netC.module.state_dict(),
                        'optimizer_state_dict': optimizerC.state_dict(),
                        'loss': test_losses
                    }
                    torch.save(save_dict, '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
                    torch.save(save_dict, '%s/%s_train_latest' % (opt.checkpoints_path, opt.name))

            if cur_iter % opt.freq_mesh == 0 and cur_iter != 0 and not opt.no_mesh_recon:          
                with torch.no_grad():
                    set_eval()
                    print('generate mesh (train) ...')
                    random.seed(1)
                    train_dataset.is_train = False
                    data_idxs = random.sample(list(range(len(train_dataset))), k=opt.num_gen_mesh_test)
                    for data_idx in tqdm(data_idxs):
                        train_data = train_dataset[data_idx]
                        save_path = '%s/%s/train_eval_epoch%d_%s_%d_%d.obj' % (
                            opt.results_path, opt.name, epoch, train_data['name'], train_data['vid'], train_data['pid'])
                        gen_mesh(opt.resolution, netC if not multi_gpu else netC.module, cuda, train_data, save_path, ls_thresh)
                    train_dataset.is_train = True

                    print('generate mesh (test) ...')
                    random.seed(1)
                    data_idxs = random.sample(list(range(len(test_dataset))), k=opt.num_gen_mesh_test)
                    for data_idx in tqdm(data_idxs):
                        test_data = test_dataset[data_idx]
                        save_path = '%s/%s/test_eval_epoch%d_%s_%d_%d.obj' % (
                            opt.results_path, opt.name, epoch, test_data['name'], test_data['vid'], test_data['pid'])
                        gen_mesh(opt.resolution, netC if not multi_gpu else netC.module, cuda, test_data, save_path, ls_thresh)
                    set_train()
                
            iter_data_time = time.time()
            cur_iter += 1

def trainerWrapper(args=None):
    opt = parser.parse(args)
    train(opt)

if __name__ == '__main__':
    trainerWrapper()