import submitit
from .train import trainerWrapper
from .test import evalWrapper

base_cmd =['--dataroot','./../../data/hf_human_big','--dataset', 'renderppl',
            '--random_flip', '--random_scale', '--random_trans', '--random_rotate', '--random_bg',
            '--linear_anneal_sigma', 
            '--norm', 'group', '--vol_norm', 'group',
            '--num_threads', '40', '--vol_ch_in', '32']

executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)
executor.update_parameters(timeout_min=72*60, gpus_per_node=4, cpus_per_task=40, partition="priority", name='wildPIFu', comment='cvpr deadline')  # timeout in min
# executor.update_parameters(timeout_min=2*60, gpus_per_node=1, cpus_per_task=10, partition="uninterrupted", name='wildPIFu')  # timeout in min


# cmd = base_cmd + ['--name','cluster_upper_1010_bce_longarm_mask_imnorm_g0.65', '--sampling_mode', 'uniform_sigma_arm_mask_aneal', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '400.0', '--occ_loss_type', 'bce', '--occ_gamma', '0.65', '--imfeat_norm']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_upper_1010_bce_longarm_face_mask_imnorm_g0.65', '--sampling_mode', 'uniform_sigma_arm_mask_face_aneal', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '400.0', '--occ_loss_type', 'bce', '--occ_gamma', '0.65', '--imfeat_norm']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_face_1020', '--crop_type', 'face', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '400.0', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_upperbody_1020', '--crop_type', 'upperbody', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '400.0', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_fullbody_1020', '--crop_type', 'fullbody', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '400.0', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--resolution', '256', '--name','test', '--crop_type', 'face', '--freq_mesh', '100', '--freq_save_ply', '100', '--mask_ratio', '0.1', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '200.0', '--occ_loss_type', 'bce']
# trainerWrapper(cmd)

# cmd = base_cmd + ['--name','cluster_face_1020_fixed', '--crop_type', 'face', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '200.0', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_upperbody_1020_fixed', '--crop_type', 'upperbody', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '200.0', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_fullbody_1020_fixed', '--crop_type', 'fullbody', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '200.0', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','test', '--crop_type', 'upperbody', '--resolution', '128', '--freq_mesh', '200', '--mask_ratio', '0.2', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '200.0', '--occ_loss_type', 'bce', '--mlp_norm', 'none']
# trainerWrapper(cmd)

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '20.0', '--sigma_min', '2.0', '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'batch']

# cmd = cmd + ['--name','cluster_face_1021_batch', '--crop_type', 'face']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name','cluster_upperbody_1021_batch', '--crop_type', 'upperbody']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name','cluster_fullbody_1021_batch', '--crop_type', 'fullbody']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'none']

# cmd = cmd + ['--name', 'cluster_face_1021_nonorm', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '0.5', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_upperbody_1021_nonorm', '--crop_type', 'upperbody', '--sigma_max', '10.0', '--sigma_min', '1.0', '--load_netG_checkpoint_path', './checkpoints/cluster_upperbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# # cmd = cmd + ['--name', 'cluster_fullbody_1021_nonorm', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '2.0', ]
# # job = executor.submit(trainerWrapper, cmd)  
# # print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'none',\
#                 '--sampling_otf', '--num_sample_surface', '4000', '--num_sample_inout', '6000', \
#                 '--num_iter', '100000', '--schedule', '60000', '--learning_rate', '1e-4', '--finetune']

# # cmd = cmd + ['--name', 'test', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sigma_surface', '2.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# # trainerWrapper(cmd)  

# cmd = cmd + ['--name', 'cluster_face_1022_nonorm_wsurface', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sigma_surface', '2.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_upperbody_1022_nonorm_wsurface', '--crop_type', 'upperbody', '--sigma_max', '8.0', '--sigma_min', '8.0', '--sigma_surface', '3.0', '--load_netG_checkpoint_path', './checkpoints/cluster_upperbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_fullbody_1022_nonorm_wsurface', '--crop_type', 'fullbody', '--sigma_max', '10.0', '--sigma_min', '10.0', '--sigma_surface', '3.0', '--load_netG_checkpoint_path', './checkpoints/cluster_fullbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# for nml in [1.0, 1e-1, 1e-2, 1e-3]:
#     cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                     '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'none',\
#                     '--num_sample_normal', '4000', '--num_sample_inout', '6000', \
#                     '--num_iter', '100000', '--schedule', '60000', '--learning_rate', '1e-4', '--finetune', '--nml_loss_type', 'l1']

#     cmd += ['--lambda_nml', '%f' % nml] 

#     cmd = cmd + ['--name', 'cluster_face_1023_nonorm_l1_wnormal%e' % nml, '--crop_type', 'face', '--sigma_max', '3.0', '--sigma_min', '3.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
#     job = executor.submit(trainerWrapper, cmd)  
#     print(job.job_id)  # ID of your job

#     cmd = cmd + ['--name', 'cluster_upperbody_1023_nonorm_l1_wnormal%e' % nml, '--crop_type', 'upperbody', '--sigma_max', '3.0', '--sigma_min', '3.0', '--load_netG_checkpoint_path', './checkpoints/cluster_upperbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
#     job = executor.submit(trainerWrapper, cmd)  
#     print(job.job_id)  # ID of your job

#     cmd = cmd + ['--name', 'cluster_fullbody_1023_nonorm_l1_wnormal%e' % nml, '--crop_type', 'fullbody', '--sigma_max', '5.0', '--sigma_min', '5.0', '--load_netG_checkpoint_path', './checkpoints/cluster_fullbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
#     job = executor.submit(trainerWrapper, cmd)  
#     print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'none',\
#                 '--num_sample_normal', '4000', '--num_sample_inout', '6000', \
#                 '--num_iter', '100000', '--schedule', '60000', '--learning_rate', '1e-4', '--finetune', '--nml_loss_type', 'l1']

# cmd += ['--lambda_nml', '1e0'] 

# cmd = cmd + ['--name', 'test', '--crop_type', 'face', '--sigma_max', '3.0', '--sigma_min', '3.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
# trainerWrapper(cmd)

# for opt in ['0.2', '0.4', '0.6']:
#     cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                     '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'none',\
#                     '--sampling_otf', '--num_sample_inout', '8000', \
#                     '--num_iter', '200000', '--schedule', '100000', '150000', '--learning_rate', '1e-3', '--finetune']
#     cmd += ['--uniform_ratio', opt]
    # cmd = cmd + ['--name', 'test', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sigma_surface', '2.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
    # trainerWrapper(cmd)  

    # cmd = cmd + ['--name', 'test', '--crop_type', 'face', '--nml_loss_type', 'mse', '--lambda_nml', '0.1', '--num_sample_normal', '4000', '--sigma_max', '2.0', '--sigma_min', '2.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
    # trainerWrapper(cmd)
    # job = executor.submit(trainerWrapper, cmd)  
    # print(job.job_id)  # ID of your job

    # cmd = cmd + ['--name', 'cluster_face_1023_nonorm_long_wsurface_u%s' % opt, '--crop_type', 'face', '--sigma_max', '2.0', '--sigma_min', '2.0', '--load_netG_checkpoint_path', './checkpoints/cluster_face_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
    # job = executor.submit(trainerWrapper, cmd)  
    # print(job.job_id)  # ID of your job

    # cmd = cmd + ['--name', 'cluster_upperbody_1023_nonorm_long_wsurface_u%s' % opt, '--crop_type', 'upperbody', '--sigma_max', '3.0', '--sigma_min', '3.0', '--load_netG_checkpoint_path', './checkpoints/cluster_upperbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
    # job = executor.submit(trainerWrapper, cmd)  
    # print(job.job_id)  # ID of your job

    # cmd = cmd + ['--name', 'cluster_fullbody_1023_nonorm_long_wsurface_u%s' % opt, '--crop_type', 'fullbody', '--sigma_max', '5.0', '--sigma_min', '5.0', '--load_netG_checkpoint_path', './checkpoints/cluster_fullbody_1021_nonorm_img.hg.group.4.2.256_wbg1_s2.20_train_latest']
    # job = executor.submit(trainerWrapper, cmd)  
    # print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--occ_loss_type', 'mse', '--mask_ratio', '0.2', '--mlp_norm', 'none',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '4000', '--num_sample_inout', '4000', \
#                 '--num_iter', '500000', '--schedule', '300000', '400000', '--learning_rate', '1e-3', '--resolution', '256']

# cmd = cmd + ['--name', 'test', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '2.0', '--sigma_surface', '2.0']
# trainerWrapper(cmd)  

# cmd = cmd + ['--name', 'cluster_face_1023_longlong_nonorm_wsurface_mse', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '2.0', '--sigma_surface', '2.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_upperbody_1023_longlong_nonorm_wsurface_mse', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'upperbody', '--sigma_max', '10.0', '--sigma_min', '3.0', '--sigma_surface', '3.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_fullbody_1023_longlong_nonorm_wsurface_mse', '--dataroot','./../../data/hf_human_big', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '3.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job


# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--occ_loss_type', 'bce', '--mask_ratio', '0.2', '--mlp_norm', 'none',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '6000', '--num_sample_inout', '2000', \
#                 '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256']

# cmd = cmd + ['--name', 'test', '--dataroot','./../../data/hf_human_big', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0']
# trainerWrapper(cmd)  

# cmd = cmd + ['--name', 'cluster_face_1024_facecenter', '--dataroot','./../../data/hf_human_face', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '2.0', '--sigma_surface', '5.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_face_1024', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '2.0', '--sigma_surface', '5.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_upperbody_1024', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'upperbody', '--sigma_max', '10.0', '--sigma_min', '3.0', '--sigma_surface', '8.0', '--continue_train']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_fullbody_1024', '--dataroot','./../../data/hf_human_big', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_face_1024_facecenter_fixgamma', '--dataroot','./../../data/hf_human_face', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '2.0', '--sigma_surface', '5.0', '--occ_gamma', '0.5']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_face_1024_fixgamma', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'face', '--sigma_max', '5.0', '--sigma_min', '2.0', '--sigma_surface', '5.0', '--occ_gamma', '0.5']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_upperbody_1024_fixgamma', '--dataroot','./../../data/hf_human_upper', '--crop_type', 'upperbody', '--sigma_max', '10.0', '--sigma_min', '3.0', '--sigma_surface', '8.0', '--occ_gamma', '0.6', '--continue_train']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_fullbody_1024_fixgamma', '--dataroot','./../../data/hf_human_big', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0', '--occ_gamma', '0.7']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_fullbody_1024_hpifu_poc', '--dataroot','./../../data/hf_human_big', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0', '--netG', 'hghpifu', '--merge_layer', '2']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_upperbody_center_1024', '--dataroot','./../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--sigma_max', '10.0', '--sigma_min', '3.0', '--sigma_surface', '8.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'test', '--dataroot','./../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--sigma_max', '10.0', '--sigma_min', '3.0', '--sigma_surface', '8.0']
# cmd = cmd + ['--name', 'cluster_fullbody_center_1025_hpifu_poc', '--dataroot','./../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0', '--netG', 'hghpifu', '--merge_layer', '2']
# trainerWrapper(cmd)

# cmd = cmd + ['--name', 'cluster_fullbody_center_1025_hpifu_poc', '--dataroot','./../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0', '--netG', 'hghpifu', '--merge_layer', '2']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = cmd + ['--name', 'cluster_fullbody_center_1025', '--dataroot','./../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--sigma_max', '20.0', '--sigma_min', '3.0', '--sigma_surface', '10.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--mask_ratio', '0.2', '--mlp_norm', 'none', '--sigma_max', '5.0', '--sigma_min', '5.0',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '8000', '--num_sample_inout', '0', \
#                 '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256']

# cmd1 = cmd + ['--name', 'cluster_upperbody_bce_1104', '--dataroot', './../../data/hf_human_upper', '--crop_type', 'upperbody', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_upperbody_mse_1104', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--occ_loss_type', 'mse']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_face_bce_1105', '--dataroot', './../../data/hf_human_face', '--crop_type', 'face', '--occ_loss_type', 'bce']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_face_mse_1105', '--dataroot', './../../data/hf_human_face', '--crop_type', 'face', '--occ_loss_type', 'mse']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_upperbody_mse_1105', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--occ_loss_type', 'mse', '--random_body_chop']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_face_mse_1105', '--dataroot', './../../data/hf_human_face', '--crop_type', 'face', '--occ_loss_type', 'mse']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_fullbody_mse_1106', '--dataroot', './../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--occ_loss_type', 'mse']
# # trainerWrapper(cmd1)
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_upperbody_mse_1106', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--occ_loss_type', 'mse', '--random_body_chop']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_upperbody_mse_v2_1106', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--occ_loss_type', 'mse', '--random_body_chop']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
#                 '--z_size', '200.0', '--mask_ratio', '0.2', '--mlp_norm', 'none', '--sigma_max', '5.0', '--sigma_min', '5.0',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '8000', '--occ_loss_type', 'mse', '--loadSize', '256',\
#                 '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256', '--continue_train']

# cmd1 = cmd + ['--name', 'cluster_fullbody256_mse_1106', '--dataroot', './../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--sigma_surface', '10.0', '--num_sample_inout', '2000']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'cluster_upperbody256_mse_1106', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--random_body_chop', '--num_sample_inout', '0']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job


cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2',\
                '--z_size', '200.0', '--mask_ratio', '0.2', '--mlp_norm', 'none', '--sigma_max', '5.0', '--sigma_min', '5.0',\
                '--sampling_otf', '--sampling_parts', '--num_sample_surface', '8000', '--occ_loss_type', 'mse',\
                '--uniform_ratio', '0.2', '--num_iter', '400000', '--schedule', '300000', '350000', '--learning_rate', '1e-3', '--resolution', '256']

cmd1 = cmd + ['--name', 'cluster_upperbody256_mse_crop12_1107', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--random_body_chop', '--loadSize', '256','--num_sample_inout', '2000']
job = executor.submit(trainerWrapper, cmd1)  
print(job.job_id)  # ID of your job

cmd1 = cmd + ['--name', 'cluster_upperbody512_mse_crop12_1107', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--random_body_chop', '--loadSize', '512','--num_sample_inout', '2000']
job = executor.submit(trainerWrapper, cmd1)  
print(job.job_id)  # ID of your job

cmd1 = cmd + ['--name', 'cluster_upperbody256_v2_mse_crop12_1107', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--random_body_chop', '--loadSize', '256','--num_sample_inout', '0']
job = executor.submit(trainerWrapper, cmd1)  
print(job.job_id)  # ID of your job

cmd1 = cmd + ['--name', 'cluster_upperbody512_v2_mse_crop12_1107', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--random_body_chop', '--loadSize', '512','--num_sample_inout', '0']
job = executor.submit(trainerWrapper, cmd1)  
print(job.job_id)  # ID of your job


# eval

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job