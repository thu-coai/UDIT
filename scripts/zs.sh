#! /bin/bash

WORKING_DIR=/home/yourname/UDIT

MP_SIZE=1

NUM_GPUS_PER_WORKER=1 # number of gpus used on one node

DATA_EXT=".jsonl"
DATA_NAMES="cb-rte-anli_r1-copa-hellaswag-wsc_balance-winogrande_xl-winogrande_debiased-wic-story_cloze_2016-squad_qg-story_gen"

MASTER_PORT=1334
CKPT=${1}
SEED=10

CONFIG_PATH="${WORKING_DIR}/configs/model/t5_large_config.json"
CKPT_PATH="${WORKING_DIR}/results/t0/fp16/ft/${CKPT}"

SAVE_PATH="${WORKING_DIR}/results/t0/fp16/zs/${DATA_NAMES}/${CKPT}"

LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_fp16.json"
TOKENIZER_PATH="${WORKING_DIR}/vocab_en"

BATCH_SIZE=16


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --dev-batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${BATCH_SIZE}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-eval"
OPTS+=" --test-num 1000"
OPTS+=" --seed ${SEED}"
OPTS+=" --eval-per-prompt"

CMD="torchrun --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/train_t0.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
