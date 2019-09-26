import submitit
from .train import trainerWrapper

base_cmd =['--dataroot','./../../data/hf_human_full','--dataset', 'renderppl',
            '--num_iter', '200000', 
            '--random_flip', '--random_scale', '--random_trans',
            '--schedule', '150000', '180000', '--sampling_otf',
            '--sampling_mode', 'uniform_sigma_aneal', '--linear_anneal_sigma', 
            '--norm', 'group', '--vol_norm', 'group', '--lambda_nml', '0.0',
            '--num_sample_inout', '6000','--num_sample_normal', '3000',   
            '--num_threads', '40', '--vol_ch', '32']

executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)
executor.update_parameters(timeout_min=48*60, gpus_per_node=4, cpus_per_task=40, partition="learnfair", name='wildPIFu')  # timeout in min

cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '2', '--hg_depth', '4', '--sigma_max', '5.0', '--sigma_min', '5.0']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '2', '--hg_depth', '4', '--sigma_max', '10.0', '--sigma_min', '1.0']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'z', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '5.0', '--sigma_min', '5.0']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '1.0']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '1.0', '--random_bg']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job

cmd = base_cmd + ['--name','cluster_wpitch', '--batch_size', '8', '--sp_enc_type', 'vol_enc', '--num_stack', '4', '--hg_depth', '2', '--sigma_max', '10.0', '--sigma_min', '1.0', '--hg_dim', '128']
job = executor.submit(trainerWrapper, cmd)  
print(job.job_id)  # ID of your job