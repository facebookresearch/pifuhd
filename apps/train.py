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
from torch.utils.tensorboard import SummaryWriter

from lib.options import BaseOptions
from lib.visualizer import Visualizer
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index

opt = BaseOptions().parse()

writer = SummaryWriter(log_dir="./logs/%s" % opt.name)

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
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except:
        print('Can not create marching cubes at this time.')

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def linear_anneal_sigma(opt, cur_epoch, n_epoch):
    opt.sigma = (opt.sigma_min - opt.sigma_max) * cur_epoch / float(n_epoch - 1) + opt.sigma_max
   
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
        error_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            net.filter(image_tensor)
            net.query(points=sample_tensor, calibs=calib_tensor, labels=label_tensor)

            res = net.get_preds()
            error = net.get_error()

            IOU, prec, recall = compute_acc(res, label_tensor, thresh)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            error_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())
    
    if label is not None:
        return {
            'MSE-%s' % label: np.average(error_arr),
            'IOU-%s' % label: np.average(IOU_arr),
            'prec-%s' % label: np.average(prec_arr),
            'recall-%s' % label: np.average(recall_arr),
        }
    else:
        return {
            'MSE': np.average(error_arr),
            'IOU': np.average(IOU_arr),
            'prec': np.average(prec_arr),
            'recall': np.average(recall_arr),
        }

def train(opt):
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    vis = Visualizer(opt)

    if opt.use_tsdf:
        train_dataset = RPTSDFDataset(opt, phase='train')
        test_dataset = RPTSDFDataset(opt, phase='val')
    elif opt.sampling_otf:
        train_dataset = RPOtfDataset(opt, phase='train')
        test_dataset = RPOtfDataset(opt, phase='val')
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
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for netG...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        model_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        if os.path.exists(model_path):
            print('Resuming from ', model_path)
            netG.load_state_dict(torch.load(model_path, map_location=cuda))
        else:
            print('Error: could not find checkpoint [%s]' % model_path)
            opt.continue_train = False
            opt.resume_epoch = 0

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # training
    start_epoch = 0 if not opt.continue_train else opt.resume_epoch
    num_epoch = opt.num_iter // len(train_data_loader)
    cur_iter = start_epoch * len(train_data_loader)
    max_IOU = 0.0
    for epoch in range(start_epoch, num_epoch):
        epoch_start_time = time.time()

        if opt.linear_anneal_sigma:
            linear_anneal_sigma(opt, epoch, num_epoch)

        set_train()
        for train_idx, train_data in  enumerate(train_data_loader):
            iter_data_time = time.time()

            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

            label_tensor = train_data['labels'].to(device=cuda)

            iter_start_time = time.time()

            netG.filter(image_tensor)
            netG.query(points=sample_tensor, calibs=calib_tensor, labels=label_tensor)

            res = netG.get_preds()
            errG = netG.get_error()
            errG /= opt.num_stack

            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)
            
            print(
                'Name: {0} | Epoch: {1}/{2} | {3}/{4} | Err: {5:05f} | LR: {6:.1e} | SIG: {7:.1e} | dataT: {8:04f} | netT: {9:04f} | ETA: {10:02d}:{11:02d}'.format(
                    opt.name, epoch, num_epoch, train_idx, len(train_data_loader), errG.item(), lr, opt.sigma,
                                                                        iter_start_time - iter_data_time,
                                                                        iter_net_time - iter_start_time, int(eta // 60),
                    int(eta - 60 * (eta // 60))))
            
            if train_idx % opt.freq_save == 100 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
                torch.save(netG.state_dict(), '%s/%s_train_latest' % (opt.checkpoints_path, opt.name))

            if train_idx % opt.freq_save_ply == 0 and train_idx != 0:
                save_path = '%s/%s/test_epoch%d_idx%d.ply' % (opt.results_path, opt.name, epoch, train_idx)
                r = res[0].cpu()
                points = sample_tensor[0].transpose(0, 1).cpu()
                save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())
  
            if train_idx % opt.freq_plot == 0:
                losses = {}
                losses['train_mse'] = errG.item() / opt.num_stack
                counter_ratio = train_idx / len(train_data_loader)

                vis.plot_current_losses(epoch, counter_ratio, losses)
                writer.add_scalar('total_loss/train', losses['train_mse'], cur_iter)

            if train_idx % opt.freq_save_image == 0:
                visuals = {}
                visuals['input'] = image_tensor
                vis.display_current_results(epoch, visuals)

            iter_data_time = time.time()
            cur_iter += 1    
            lr = adjust_learning_rate(optimizerG, cur_iter, lr, opt.schedule, opt.gamma)

        ## test
        with torch.no_grad():
            set_eval()
            test_losses = {}

            if not opt.no_numel_eval:
                print('calc error (train) ...')
                train_dataset.is_train = False
                err = calc_error(opt, netG, cuda, train_dataset, 100, 'train', ls_thresh)
                train_dataset.is_train = True
                print('eval: ', ''.join(['{}: {:.6f} '.format(k, v) for k,v in err.items()]))
                test_losses.update(err)
                for k, v in err.items():
                    writer.add_scalar('%s/%s' % (k.split('-')[0],'train'), v, cur_iter)

                print('calc error (test) ...')
                err = calc_error(opt, netG, cuda, test_dataset, 360, 'test', ls_thresh)
                if err['IOU-test'] > max_IOU:
                    max_IOU = err['IOU-test']
                print('eval: ', ''.join(['{}: {:.6f} '.format(k, v) for k,v in err.items()]), 'bestIOU: %.3f' % max_IOU)
                test_losses.update(err)
                for k, v in err.items():
                    writer.add_scalar('%s/%s' % (k.split('-')[0],'test'), v, cur_iter)

                vis.plot_current_test_losses(epoch, 0, test_losses)

            if not opt.no_mesh_recon:          
                print('generate mesh (train) ...')
                random.seed(1)
                data_idxs = random.sample(list(range(len(train_dataset))), k=opt.num_gen_mesh_test)
                for data_idx in data_idxs:
                    train_data = train_dataset[data_idx]
                    save_path = '%s/%s/train_eval_epoch%d_%s_%d_%d.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'], train_data['vid'], train_data['pid'])
                    gen_mesh(opt.resolution, netG, cuda, train_data, save_path, ls_thresh)

                print('generate mesh (test) ...')
                random.seed(1)
                data_idxs = random.sample(list(range(len(test_dataset))), k=opt.num_gen_mesh_test)
                for data_idx in data_idxs:
                    test_data = test_dataset[data_idx]
                    save_path = '%s/%s/test_eval_epoch%d_%s_%d_%d.obj' % (
                        opt.results_path, opt.name, epoch, test_data['name'], test_data['vid'], test_data['pid'])
                    gen_mesh(opt.resolution, netG, cuda, test_data, save_path, ls_thresh)

if __name__ == '__main__':
    train(opt)
  