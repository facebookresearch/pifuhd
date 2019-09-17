set -ex

# Dataset
DATASET='renderppl'
DATAROOT='/home/shunsukesaito/data/hf_human'

# Training
NAME='devfair'

SAMPLE_MODE='sigma3_uniform'
SIGMA_MAX=3.0

NUM_SAMPLE=50000

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