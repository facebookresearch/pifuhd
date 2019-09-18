import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataset', type=str, default='renderppl', help='dataset name')
        g_data.add_argument('--dataroot', type=str, default='./data',
                            help='path to images (data folder)')

        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')
        g_data.add_argument('--classes', default='chair', type=str, help='class to use')  # only for shapenet
        g_data.add_argument('--with_mask', action='store_true', help='applying silhouette in shapenet training')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')
        g_exp.add_argument('--mode', type=str, default='inout', help='inout || color')
        g_exp.add_argument('--use_tsdf', action='store_true', help='use tsdf instead of occupancy')

        g_exp.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
        g_exp.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        g_train.add_argument('--batch_size', type=int, default=32, help='input batch size')
        g_train.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        g_train.add_argument('--display_port', type=int, default=8100, help='port of visdom')
        g_train.add_argument('--display_id', type=int, default=0, help='port of visdom')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_iter', type=int, default=30000, help='num iterations to train')
        g_train.add_argument('--freq_plot', type=int, default=500, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=3000, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_mesh', type=int, default=4000, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=1000, help='freqency of the save ply')
        g_train.add_argument('--freq_save_image', type=int, default=100, help='freqency of the save input image')
        g_train.add_argument('--resume_epoch', type=int, default=0, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=512, help='# of grid in mesh reconstruction')
        g_test.add_argument('--no_numel_eval', action='store_true', help='no numerical evaluation')
        g_test.add_argument('--no_mesh_recon', action='store_true', help='no mesh reconstruction')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--num_sample', type=int, default=10000, help='# of sampling points')
        g_sample.add_argument('--num_sample_inout', type=int, default=0, help='# of sampling points')
        g_sample.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')
        g_sample.add_argument('--num_sample_normal', type=int, default=0, help='# of sampling points')
        g_sample.add_argument('--num_pts_dic', type=int, default=5, help='# of pts dic you load')

        g_sample.add_argument('--max_pitch', type=int, default=45, help='max pitch angle')
        g_sample.add_argument('--mean_pitch', type=int, default=0, help='pitch center')
        g_sample.add_argument('--sampling_otf', action='store_true', help='Sampling on the fly')
        g_sample.add_argument('--sampling_mode', type=str, default='sigma', help='Sampling file name.')
        g_sample.add_argument('--linear_anneal_sigma', action='store_true', help='linear annealing of sigma')
        g_sample.add_argument('--sigma_max', type=float, default=0.0, help='maximum sigma for sampling')
        g_sample.add_argument('--sigma_min', type=float, default=0.0, help='minimum sigma for sampling')
        g_sample.add_argument('--sigma', type=float, default=1.0, help='sigma for sampling')

        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='batch',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--color_type', type=str, default=None, help='where to add volume feature [ feature | input ]')
        

        # Image filter General
        g_model.add_argument('--netG', type=str, default='piximp', help='piximp | fanimp')
        g_model.add_argument('--netC', type=str, default='resimp', help='resimp | fanimp')
        g_model.add_argument('--enc_dim', nargs='+', default=[3, 8, 16, 32, 64, 128], type=int,
                             help='# of dimensions of encoder')

        # hgimp specific
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--hg_depth', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hg_dim', type=int, default=256, help='256 | 512')
        g_model.add_argument('--use_sigmoid', action='store_true', help='use sigmoid function in fanimp or not')

        # volumetric encoder
        g_model.add_argument('--sp_no_pifu', action='store_true', help='cut fcn feature for debug')
        g_model.add_argument('--sp_enc_type', type=str, default='z', help='spatial encoding [ z | vol_enc ]')
        g_model.add_argument('--vol_net', type=str, default='unet', help='[ unet | hg ]')
        g_model.add_argument('--vol_norm', type=str, default='batch', help='normalization for volume branch')
        g_model.add_argument('--vol_ch', type=int, default=16, help='channel size for volume branch')
        g_model.add_argument('--vol_hg_depth', type=int, default=2, help='depth of hourglass in volume branch')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')
        g_model.add_argument('--mlp_dim_color', nargs='+', default=[1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')
        g_model.add_argument('--mlp_res_layers', nargs='+', default=[1,2,3,4], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')
        g_model.add_argument('--begin_mlp', type=int, default=0, help='# of skip mlp layer')
        g_model.add_argument('--skip_z', action='store_true',
                             help='do not use z as the feature.')

        # for train
        parser.add_argument('--random_flip', action='store_true', help='if random flip')
        parser.add_argument('--random_trans', action='store_true', help='if random flip')
        parser.add_argument('--random_scale', action='store_true', help='if random flip')
        parser.add_argument('--schedule', type=int, nargs='+', default=[10, 15],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--lambda_nml', type=float, default=1, help='weight of normal loss')
        parser.add_argument('--lambda_inout', type=float, default=1, help='weight of input loss')
        parser.add_argument('--lambda_color', type=float, default=1, help='weight of color loss')
        parser.add_argument('--lambda_gan', type=float, default=1, help='weight of color gan loss')
        parser.add_argument('--nml_loss_type', type=str, default='mse', help='mse | l1')
        parser.add_argument('--color_loss_type', type=str, default='mse', help='mse | l1')
        parser.add_argument('--no_finetune', action='store_true', help='fine tuning netG in training C')
        parser.add_argument('--random_bg', action='store_true', help='using random background')

        parser.add_argument('--color_gan', action='store_true', help='use pointnet discriminator in training C')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=5,
                            help='how many meshes to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask')
        parser.add_argument('--img_path', type=str, help='path for input image')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_gry', type=float, default=0.0, help='augmentation gray scale')
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()

        opt.sigma = opt.sigma_max

        if len(opt.mlp_res_layers) == 1 and opt.mlp_res_layers[0] < 1:
            opt.mlp_res_layers = []
        
        if opt.sp_enc_type == 'vol_enc':
            opt.name = '%s_p%d.%d_%s%d_np%d_s%1.f.%1.f' % (opt.name, opt.mean_pitch, opt.max_pitch, opt.vol_net, opt.vol_ch, int(opt.sp_no_pifu), opt.sigma_min, opt.sigma_max)
            opt.mlp_dim = [opt.vol_ch if opt.sp_no_pifu else opt.vol_ch + opt.hg_dim] + opt.mlp_dim
        else:
            opt.name = '%s_p%d.%d' % (opt.name, opt.mean_pitch, opt.max_pitch)
            opt.mlp_dim = [opt.hg_dim + 1] + opt.mlp_dim

        return opt
