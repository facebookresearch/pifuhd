set -ex

# Dataset
DATASET='renderppl'
DATAROOT='./../../data/hf_human_full'

# Training
NAME='devfair'

SAMPLE_MODE='uniform_sigma5'
SIGMA_MAX=5.0

NUM_SAMPLE=500000

CHECKPOINTS_PATH='./checkpoints'
RESULTS_PATH='./results'

NETG='hgpifu'

# command
python ./apps/generate_points.py \
    --dataroot ${DATAROOT} \
    --dataset ${DATASET} \
    --name ${NAME} \
    --results_path ${RESULTS_PATH} \
    --num_sample_inout ${NUM_SAMPLE} \
    --sampling_mode ${SAMPLE_MODE} \
    --sampling_otf \
    --sigma_max ${SIGMA_MAX}