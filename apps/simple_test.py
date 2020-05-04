# import submitit
from .train import trainerWrapper
from .recon_eval import reconWrapper
from .recon_mr import reconWrapper as reconWrapperMR
from .recon_mr_eval import reconWrapper as reconWrapperMREval

# executor = submitit.AutoExecutor(folder="cluster_log")  # submission interface (logs are dumped in the folder)
# executor.update_parameters(timeout_min=2*60, gpus_per_node=4, cpus_per_task=40, partition="dev", name='wildPIFu', comment='cvpr deadline')  # timeout in min

###############################################################################################
##                   Setting
###############################################################################################
input_path = '/private/home/hjoo/pifu_test/input'              #,where your jpg (raw image) and json (openpose output) files exist
output_path = '/private/home/hjoo/pifu_test/output'             #, where reconstruction outputs are saved
checkpoint_path= '/private/home/hjoo/data/pifuhd/checkpoints/ours_final_train_latest''

###############################################################################################
##                   Upper PIFu
###############################################################################################

resolution = '512'

start_id = -1
end_id = -1
cmd = ['--dataroot', input_path, '--results_path', output_path,\
       '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path', \
       checkpoint_path,\
       '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
reconWrapperMR(cmd)

