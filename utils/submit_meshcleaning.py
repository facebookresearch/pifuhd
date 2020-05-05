import os
import submitit
from connected_comp import meshcleaning


executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)

rootDir = '/private/home/hjoo/codes/wildpifu/video_cand'
job_nums = 10       #Num of jobs to process mesh cleaning per each sequence


SUBDIRNAME ='mr_fullbody_no_nml_hg_fp0_1112/recon'  #This may need to be changed based on your model
seqnames = os.listdir(rootDir)

# seqnames = ['shun_1_output', 'shun_2_output']
for seqname in seqnames:

    if "_output" not in seqname:
        continue

    if 'shun' in seqname:
        continue

    executor.update_parameters(timeout_min=4320, gpus_per_node=1, cpus_per_task=10, partition="priority", name=seqname, comment='cvpr oral supp May 13th')  # timeout in min
    
    filepath = os.path.join(rootDir,seqname,SUBDIRNAME)
    if os.path.exists(filepath)==False:
        continue
    
    print(f"checking: {filepath}")

    numobj = len(os.listdir(filepath))
    print(f"objnum: {numobj}")

    for j in range(job_nums):
        # meshcleaning(filepath, job_nums, j)
        job = executor.submit(meshcleaning, filepath, job_nums, j)
