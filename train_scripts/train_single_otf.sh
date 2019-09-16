set -ex

# Dataset
DATASET='renderppl'
DATAROOT='/home/shunsukesaito/data/hf_human'

# Training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+10))
DISPLAY_PORT=8097
NAME='pifu_debug'

SAMPLE_MODE='sigma/uniform'
SIGMA_MAX=5.0

NUM_ITER=100000
NUM_VIEWS=1
BATCH_SIZE=3
ENC_DIM='3 8 16 32 64 128'
MLP_DIM='1024 512 256 128 1'
MLP_DIM_COLOR='513 1024 512 256 128 3'
MLP_RES_LAYERS='2'
NUM_SAMPLE=5000
NUM_THREADS=1
VOL_RES=256

AUG_BRI=0.2
AUG_CON=0.2
AUG_SAT=0.08
AUG_HUE=0.08
AUG_GRY=0.1

MAX_PITCH=45

SPENC_TYPE='z'
VOL_NET='unet'
VOL_CH=32
VOL_HG_DEPTH=3

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
    --num_iter ${NUM_ITER} \
    --batch_size ${BATCH_SIZE} \
    --enc_dim ${ENC_DIM} \
    --mlp_dim ${MLP_DIM} \
    --mlp_dim_color ${MLP_DIM_COLOR} \
    --mlp_res_layers ${MLP_RES_LAYERS} \
    --checkpoints_path ${CHECKPOINTS_PATH} \
    --results_path ${RESULTS_PATH} \
    --num_sample_inout ${NUM_SAMPLE} \
    --num_threads ${NUM_THREADS} \
    --random_flip \
    --netG ${NETG} \
    --num_stack 2 \
    --hg_depth 2 \
    --schedule 50000 75000 \
    --resolution ${VOL_RES} \
    --hg_down 'ave_pool' \
    --random_scale \
    --random_trans \
    --random_flip \
    --max_pitch ${MAX_PITCH} \
    --sampling_mode ${SAMPLE_MODE} \
    --sampling_otf \
    --sigma_max ${SIGMA_MAX} \
    --num_pts_dic 5 \
    --norm 'group' \
    --freq_save_ply 5000 \
    --aug_bri ${AUG_BRI} \
    --aug_con ${AUG_CON} \
    --aug_sat ${AUG_SAT} \
    --aug_hue ${AUG_HUE} \
    --sp_enc_type ${SPENC_TYPE}