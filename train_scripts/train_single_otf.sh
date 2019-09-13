set -ex

# Dataset
DATASET='renderppl'
DATAROOT='/home/shunsukesaito/data/hf_human'

# Training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+10))
DISPLAY_PORT=8097
NAME='otf_sigma5_uniform_group'

SAMPLE_MODE='sigma/uniform'
PROB_MIN=0.6
PROB_MAX=1.0
SIGMA_MIN=1.0
SIGMA_MAX=5.0
THICK_MIN=0.005
THICK_MAX=0.02

NUM_EPOCH=200
NUM_VIEWS=1
BATCH_SIZE=3
ENC_DIM='3 8 16 32 64 128'
MLP_DIM='257 1024 512 256 128 1'
MLP_DIM_COLOR='513 1024 512 256 128 3'
NUM_SAMPLE=5000
NUM_THREADS=1
VOL_RES=128

AUG_BRI=0.2
AUG_CON=0.2
AUG_SAT=0.08
AUG_HUE=0.08
AUG_GRY=0.1

CHECKPOINTS_PATH='./checkpoints'
RESULTS_PATH='./results'

NETG='hgpifu'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./apps/train.py \
    --display_id ${DISPLAY_ID} \
    --display_port ${DISPLAY_PORT} \
    --dataroot ${DATAROOT} \
    --dataset ${DATASET} \
    --name ${NAME} \
    --num_epoch ${NUM_EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --enc_dim ${ENC_DIM} \
    --mlp_dim ${MLP_DIM} \
    --mlp_dim_color ${MLP_DIM_COLOR} \
    --checkpoints_path ${CHECKPOINTS_PATH} \
    --results_path ${RESULTS_PATH} \
    --num_sample_inout ${NUM_SAMPLE} \
    --num_threads ${NUM_THREADS} \
    --random_flip \
    --netG ${NETG} \
    --num_stack 2 \
    --hg_depth 2 \
    --schedule 120 180 \
    --resolution ${VOL_RES} \
    --hg_down 'ave_pool' \
    --random_scale \
    --random_trans \
    --random_flip \
    --sampling_mode ${SAMPLE_MODE} \
    --sampling_otf \
    --prob_min ${PROB_MIN} \
    --prob_max ${PROB_MAX} \
    --thick_min ${THICK_MIN} \
    --thick_max ${THICK_MAX} \
    --sigma_min ${SIGMA_MIN} \
    --sigma_max ${SIGMA_MAX} \
    --num_pts_dic 5 \
    --norm 'group' \
    --freq_save_ply 5000 \
    --aug_bri ${AUG_BRI} \
    --aug_con ${AUG_CON} \
    --aug_sat ${AUG_SAT} \
    --aug_hue ${AUG_HUE} \
    --resume_epoch 10 \
    --continue_train

