#!/bin/bash
set -Eeuo pipefail
source `dirname $0`/../config

SRC=de
TGT=en

DATA_DIR=${DATA_HOME}/de-text-clean-durations-en-phones-durations
MODEL_DIR=${MODELS_HOME}/baseline_interleaved
mkdir -p ${MODEL_DIR}

if [[ "$(which sockeye-train)" == "" ]]; then
    echo "sockeye-train not found! Check if you're using the right Python environment."
    exit 1
fi

# Prepare data
if [[ ! -f ${MODEL_DIR}/prepared_data/shard.00000 ]]; then
    echo "Preparing data"
    sockeye-prepare-data \
        --source ${DATA_DIR}/train.${SRC} \
        --target ${DATA_DIR}/train.${TGT} \
        --max-seq-len "150:400" \
        --output ${MODEL_DIR}/prepared_data
else
    echo "Data already prepared in ${MODEL_DIR}/prepared_data"
fi

# Set the number of GPUs for distributed training
# Adjust BATCH_SIZE and UPDATE_INTERVAL according to your GPU situation.
# For example, if you change N_GPU to 2, you should set update-interval to 4 to have the same effective batch size
N_GPU=1
BATCH_SIZE=4096
UPDATE_INTERVAL=8

# Run training
torchrun --no_python --nproc_per_node ${N_GPU} \
sockeye-train \
    --prepared-data ${MODEL_DIR}/prepared_data \
    --validation-source ${DATA_DIR}/valid.${SRC} \
    --validation-target ${DATA_DIR}/valid.${TGT} \
    --output ${MODEL_DIR}/model \
    --learning-rate-scheduler-type inv-sqrt-decay \
    --learning-rate-warmup 4000 \
    --embed-dropout 0.3:0.3 \
    --transformer-dropout-prepost 0.3:0.3 \
    --transformer-dropout-act 0.3:0.3 \
    --weight-tying-type trg_softmax \
    --weight-decay 0.0001 \
    --label-smoothing 0.1 \
    --label-smoothing-impl torch \
    --optimizer-betas 0.9:0.98 \
    --initial-learning-rate 0.031625 \
    --batch-size ${BATCH_SIZE} \
    --batch-type max-word \
    --update-interval ${UPDATE_INTERVAL} \
    --max-num-epochs 300 \
    --checkpoint-interval 10000 \
    --decode-and-evaluate 100 \
    --stop-training-on-decoder-failure \
    --seed 42 \
    --quiet-secondary-workers \
    --dist
