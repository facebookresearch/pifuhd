import submitit
from .train import trainerWrapper
from .train_wnml import trainerWrapper as trainerWrapperNML
from .train_mr import trainerWrapper as trainerWrapperMR

from .recon import reconWrapper


base_cmd =['--dataroot','/private/home/hjoo/data/pifuhd/CVPR2020/data/pifu_data', '--dataset', 'renderppl',
            '--random_flip', '--random_scale', '--random_trans', '--random_rotate', '--random_bg',
            '--linear_anneal_sigma', '--norm', 'group', '--num_threads', '40', '--crop_type', 'fullbody',
            '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2']
# VS code debugging
# base_cmd =['--dataroot','/private/home/hjoo/data/pifuhd/CVPR2020/data/pifu_data', '--dataset', 'renderppl',
#             '--random_flip', '--random_scale', '--random_trans', '--random_rotate', '--random_bg', #'--continue_train',
#             '--linear_anneal_sigma', '--norm', 'group', '--num_threads', '40', '--crop_type', 'fullbody',
#             '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2']



executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)
executor.update_parameters(timeout_min=72*60, gpus_per_node=4, cpus_per_task=40, partition="dev", name='wildPIFu', comment='cvpr deadline')  # timeout in min

###############################################################################################
##                   Lower PIFu
###############################################################################################

# plain
cmd = base_cmd + ['--name', 'lower_pifu_pretrained', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sigma_surface', '10.0',\
                '--batch_size', '8', '--num_stack', '4', '--hg_depth', '2', '--mlp_norm', 'none',\
                '--sampling_otf', '--sampling_parts', '--num_sample_surface', '6000', '--num_sample_inout', '2000', \
                '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256']

cmd = cmd + ['--load_netG_checkpoint_path', '/private/home/hjoo/data/pifuhd/checkpoints/lower_pifu_train_latest',
            '--checkpoints_path','checkpoint_pretrained','--finetune']


# for local
trainerWrapper(cmd)

# # for server
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job


# ###############################################################################################
# ##                   Lower PIFu With Normal Conditioning
# ###############################################################################################

# # separate version
# cmd = base_cmd + ['--name', 'lower_pifu_wnml_separate', \
#                 '--batch_size', '4', '--num_stack', '4', '--hg_depth', '2', '--sigma_surface', '10.0',\
#                 '--mlp_norm', 'none', '--sigma_max', '5.0', '--sigma_min', '5.0',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '6000', '--num_sample_inout', '2000', \
#                 '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256',\
#                 '--use_back_normal', '--load_netB_checkpoint_path', '/private/home/shunsukesaito/dev/pix2pixHD/checkpoints/f2b512_crop_r2n/latest_net_G.pth',\
#                 '--use_front_normal', '--load_netF_checkpoint_path', '/private/home/shunsukesaito/dev/pix2pixHD/checkpoints/f2f512_crop_r2n/latest_net_G.pth']

# job = executor.submit(trainerWrapperNML, cmd)  
# print(job.job_id)  # ID of your job

# # all in one version
# cmd = base_cmd + ['--name', 'lower_pifu_wnml_allinone',\
#                 '--batch_size', '2', '--num_stack', '4', '--hg_depth', '2', '--sigma_surface', '10.0',\
#                 '--mlp_norm', 'none', '--sigma_max', '5.0', '--sigma_min', '5.0',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '6000', '--num_sample_inout', '2000', \
#                 '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256',\
#                 '--use_back_normal', '--load_netFB_checkpoint_path', '/private/home/shunsukesaito/dev/pix2pixHD/checkpoints/f2b512_crop_allinone/latest_net_G.pth',\
#                 '--use_front_normal', '--use_aio_normal']

# job = executor.submit(trainerWrapperNML, cmd)  
# print(job.job_id)  # ID of your job

# ###############################################################################################
# ##                   Multi-Level PIFu
# ###############################################################################################


# # final model
# cmd = base_cmd + ['--name', 'ours_wnml',
#                 '--batch_size', '2', '--num_local', '2', '--num_stack', '1', '--hg_depth', '4',
#                 '--mlp_norm', 'none', '--sigma_max', '3.0', '--sigma_min', '3.0', '--sampling_mode', 'sigma3_uniform',
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '8000', '--num_sample_inout', '0', 
#                 '--uniform_ratio', '0.2', '--num_iter', '200000', '--schedule', '100000', '150000', '--learning_rate', '1e-3', '--resolution', '512',
#                 '--load_netG_checkpoint_path', '/private/home/shunsukesaito/CVPR2020/checkpoints/lower_pifu_wnml_train_latest',
#                 '--loadSizeBig', '1024', '--loadSizeLocal', '512', '--hg_dim', '16', '--mlp_dim', '272', '512', '256', '128', '1', 
#                 '--mlp_res_layers', '1', '2', '--merge_layer', '2']

# job = executor.submit(trainerWrapperMR, cmd)  
# print(job.job_id)  # ID of your job

# ###############################################################################################
# ##                   Ablation Study
# ###############################################################################################

# # for ablation study, trainining resnet 
# cmd = base_cmd + ['--name', 'ablation_resnet', '--batch_size', '2', '--netG', 'resnet', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--mask_ratio', '0.2', '--mlp_norm', 'batch', '--sigma_max', '5.0', '--sigma_min', '5.0',\
#                 '--sampling_parts', '--num_sample_surface', '8000', '--num_sample_inout', '0', '--occ_loss_type', 'bce',\
#                 '--uniform_ratio', '0.2', '--num_iter', '200000', '--schedule', '100000', '150000', '--learning_rate', '1e-3', '--resolution', '256',\
#                 '--loadSize', '1024', '--sampling_mode', 'sigma3_uniform', '--mlp_dim', '1025', '1024', '512', '256', '128', '1']

# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# # for ablation study, training upper layer only
# cmd = base_cmd + ['--batch_size', '4', '--num_stack', '1', '--hg_depth', '4',\
#                 '--mlp_norm', 'none', '--sigma_max', '3.0', '--sigma_min', '3.0', '--sampling_mode', 'sigma3_uniform',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '8000', '--num_sample_inout', '0', \
#                 '--uniform_ratio', '0.2', '--num_iter', '300000', '--schedule', '200000', '250000', '--learning_rate', '1e-3', '--resolution', '256',
#                 '--loadSizeBig', '1024', '--loadSizeLocal', '1024', '--hg_dim', '16',
#                 '--mlp_res_layers', '1', '2', '--merge_layer', '2']

# for with_resnet in [1, 0]:
#     cmd1 = cmd + ['--name', 'fullres_ablation_resnet%d' % (with_resnet), \
#         '--num_local', '1', '--netG', 'hg_ablation_resnet' if with_resnet else 'hg_ablation']
    
#     if with_resnet:
#         cmd1 += ['--mlp_dim', '529', '512', '256', '128', '1']
#     else:
#         cmd1 += ['--mlp_dim', '17', '512', '256', '128', '1']

#     job = executor.submit(trainerWrapperMR, cmd1)  
#     print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name', 'ours_nonml', '--num_local', '1',
#                 '--batch_size', '2', '--num_stack', '1', '--hg_depth', '2',
#                 '--mlp_norm', 'none', '--sigma_max', '3.0', '--sigma_min', '3.0', '--sampling_mode', 'sigma3_uniform',
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '8000', '--num_sample_inout', '0', 
#                 '--uniform_ratio', '0.2', '--num_iter', '200000', '--schedule', '100000', '150000', '--learning_rate', '1e-3', '--resolution', '256',
#                 '--load_netG_checkpoint_path', '/private/home/shunsukesaito/CVPR2020/checkpoints/lower_pifu_train_latest',
#                 '--loadSizeBig', '1024', '--loadSizeLocal', '512', '--hg_dim', '16', '--mlp_dim', '272', '512', '256', '128', '1', 
#                 '--mlp_res_layers', '1', '2', '--merge_layer', '2']

# job = executor.submit(trainerWrapperMR, cmd)  
# print(job.job_id)  # ID of your job