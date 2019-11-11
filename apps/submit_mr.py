import submitit
from .train_mr import trainerWrapper
from .test import evalWrapper

base_cmd =['--dataroot','./../../data/hf_human_big','--dataset', 'renderppl',
            '--random_flip', '--random_scale', '--random_trans', '--random_rotate', '--random_bg',
            '--linear_anneal_sigma', 
            '--norm', 'group', '--vol_norm', 'group',
            '--num_threads', '40', '--vol_ch_in', '32']

executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)
executor.update_parameters(timeout_min=72*60, gpus_per_node=4, cpus_per_task=40, partition="priority", name='wildPIFu', comment='cvpr deadline')  # timeout in min
# executor.update_parameters(timeout_min=2*60, gpus_per_node=1, cpus_per_task=10, partition="uninterrupted", name='wildPIFu')  # timeout in min


cmd = base_cmd + ['--batch_size', '4', '--sp_enc_type', 'z', '--num_stack', '1', '--hg_depth', '4',\
                '--z_size', '200.0', '--mask_ratio', '0.2', '--mlp_norm', 'none', '--sigma_max', '3.0', '--sigma_min', '3.0',\
                '--sampling_otf', '--sampling_parts', '--num_sample_surface', '10000', '--num_sample_inout', '0', \
                '--uniform_ratio', '0.2', '--num_iter', '200000', '--schedule', '100000', '150000', '--learning_rate', '1e-3', '--resolution', '512',
                '--load_netG_checkpoint_path', 'checkpoints/cluster_fullbody_center_1025_img.hg.group.4.2.256_wbg1_s3.20_train_latest',
                '--loadSizeBig', '2048', '--loadSizeLocal', '512', '--num_local', '4', '--hg_dim', '16', '--mlp_dim', '272', '512', '256', '128', '1', 
                '--mlp_res_layers', '1', '2', '--merge_layer', '2']

# cmd1 = cmd + ['--name', 'mr_test', '--dataroot', './../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--occ_loss_type', 'mse', '--num_sample_normal', '8000', \
#     '--nml_loss_type', 'l1', '--lambda_nml', '1e-2']
# trainerWrapper(cmd1)

for fullpifu in [0, 1]:
    for nml in ['1e0', '1e-1', '1e-2', '1e-3']:
        cmd1 = cmd + ['--name', 'mr_fullbody_1106_nml%s_fp%d' % (nml, fullpifu), '--dataroot', './../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--occ_loss_type', 'mse', '--num_sample_normal', '8000', \
            '--nml_loss_type', 'l1', '--lambda_nml', nml]
        if fullpifu == 1:
            cmd1 += ['--train_full_pifu']
        job = executor.submit(trainerWrapper, cmd1)  
        print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'mr_fullbody_1106', '--dataroot', './../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--occ_loss_type', 'mse']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'mr_fullbody_fulltrain_1106', '--dataroot', './../../data/hf_human_fullbody', '--crop_type', 'fullbody', '--occ_loss_type', 'mse', '--train_full_pifu']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '1', '--hg_depth', '4',\
#                 '--z_size', '200.0', '--mask_ratio', '0.2', '--mlp_norm', 'none', '--sigma_max', '3.0', '--sigma_min', '3.0',\
#                 '--sampling_otf', '--sampling_parts', '--num_sample_surface', '10000', '--num_sample_inout', '0', \
#                 '--uniform_ratio', '0.2', '--num_iter', '200000', '--schedule', '100000', '150000', '--learning_rate', '1e-3', '--resolution', '512',
#                 '--load_netG_checkpoint_path', 'checkpoints/cluster_fullbody_center_1025_img.hg.group.4.2.256_wbg1_s3.20_train_latest',
#                 '--loadSizeBig', '2048', '--loadSizeLocal', '512', '--num_local', '4', '--hg_dim', '16', '--mlp_dim', '272', '512', '256', '128', '1', 
#                 '--mlp_res_layers', '1', '2', '--merge_layer', '2']

# cmd1 = cmd + ['--name', 'mr_upperbody', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--occ_loss_type', 'mse']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job

# cmd1 = cmd + ['--name', 'mr_upperbody_fulltrain', '--dataroot', './../../data/hf_human_upperbody', '--crop_type', 'upperbody', '--occ_loss_type', 'mse', '--train_full_pifu']
# job = executor.submit(trainerWrapper, cmd1)  
# print(job.job_id)  # ID of your job
