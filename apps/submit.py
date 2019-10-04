import submitit
from .train import trainerWrapper
from .test import evalWrapper
from .recon import reconWrapper

base_cmd =['--dataroot','./../../data/hf_human_full','--dataset', 'renderppl',
            '--num_iter', '300000', 
            '--random_flip', '--random_scale', '--random_trans',
            '--schedule', '200000', '250000', '--sampling_otf',
            '--sampling_mode', 'uniform_sigma_aneal', '--linear_anneal_sigma', 
            '--norm', 'group', '--vol_norm', 'group', '--lambda_nml', '0.0',
            '--num_sample_inout', '6000', '--num_threads', '40', '--vol_ch', '32']

executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)
executor.update_parameters(timeout_min=72*60, gpus_per_node=4, cpus_per_task=40, partition="uninterrupted", name='wildPIFu')  # timeout in min
# executor.update_parameters(timeout_min=1*60, gpus_per_node=1, cpus_per_task=10, partition="dev", name='wildPIFu')  # timeout in min

# cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0', '--random_bg']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--random_bg']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--hg_dim', '128']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--hg_dim', '64']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--hg_dim', '32']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_no_pifu', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sp_no_pifu']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_no_pifu', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sp_no_pifu']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--random_bg']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_cmp', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2',\
#      '--sigma_max', '5.0', '--sigma_min', '5.0', '--mlp_dim', '1024', '512', '256', '128', '10', '--use_compose', '--lambda_cmp_l1', '0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_cmp', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2',\
#      '--sigma_max', '5.0', '--sigma_min', '5.0', '--mlp_dim', '1024', '512', '256', '128', '10', '--random_bg',\
#      '--use_compose', '--lambda_cmp_l1', '0']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job


# cmd = base_cmd + ['--name','cluster_wpitch_batch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_ch', '1']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_simple_c', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_net', 'simple_c']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_simple_1', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_net', 'simple_1']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name', 'cluster_wpitch_hg3d', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_net', 'hg', '--random_bg', '--continue_train']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster_wpitch_long', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--num_sample_normal', '3000', '--random_bg', '--continue_train']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_long', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--num_sample_normal', '3000', '--random_bg']
# job = executor.submit(trainerWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_long_even', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--num_sample_normal', '3000', '--random_bg']
# trainerWrapper(cmd)

## eval

# cmd = base_cmd + ['--name', 'cluster', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_no_pifu', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sp_no_pifu', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_no_pifu', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--sp_no_pifu', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0', '--random_bg', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--random_bg', '--no_numel_eval']
# job = executor.submit(reconWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '2.0', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_cmp', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2',\
#      '--sigma_max', '5.0', '--sigma_min', '5.0', '--mlp_dim', '1024', '512', '256', '128', '10', '--use_compose', '--lambda_cmp_l1', '1e-3', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_batch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_ch', '1', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_simple_c', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_net', 'simple_c', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster_wpitch_simple_1', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--vol_net', 'simple_1', '--no_numel_eval']
# job = executor.submit(evalWrapper, cmd)  
# print(job.job_id)  # ID of your job

# cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0', '--random_bg']
# job = executor.submit(reconWrapper, cmd)  
# print(job.job_id)  # ID of your job

