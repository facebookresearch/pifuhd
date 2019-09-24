set -ex

# Dataset
DATASET='renderppl'
DATAROOT='/home/shunsukesaito/data/hf_human_new'

# Training
NAME='devfair'

SAMPLE_MODE='sigma1_uniform'
SIGMA_MAX=1.0

NUM_SAMPLE=100000

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
    --sigma_max ${SIGMA_MAX} \