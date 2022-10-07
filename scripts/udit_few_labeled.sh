#! /bin/bash

WORKING_DIR=/home/yourname/UDIT

MP_SIZE=1

NUM_GPUS_PER_WORKER=2 # number of gpus used on one node

DATA_EXT=".jsonl"
DATA_NAMES="MCQA-EXQA-CBQA-TC-SENT-S2T-SUM-PARA-MCQA_PSEUDO-EXQA_PSEUDO-CBQA_PSEUDO-TC_PSEUDO-SENT_PSEUDO-S2T_PSEUDO-SUM_PSEUDO-PARA_PSEUDO"

MASTER_PORT=${1-1051}
LR=${2-0.00005}
GRAD_ACC=${3-8}
SEED=${4-20}

CONFIG_PATH="${WORKING_DIR}/configs/model/t5_large_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/t5-large-lm/t5-MP1/"

DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_fp16.json"
TOKENIZER_PATH="${WORKING_DIR}/vocab_en"

BATCH_SIZE=64
DEV_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
TRAIN_ITER=-1
EPOCHS=50
TRAIN_NUM=10000
FEW_NUM=100
FEW_DATA_NAMES="MCQA-EXQA-CBQA-TC-SENT-S2T-SUM-PARA"

SAVE_PATH="${WORKING_DIR}/results/t0/fp16/ft/${DATA_NAMES}_few_${FEW_DATA_NAMES}/lr${LR}_G${GRAD_ACC}_bs${NUM_GPUS_PER_WORKER}_${BATCH_SIZE}_num${TRAIN_NUM}/seed${SEED}/"
LOG_FILE="${SAVE_PATH}/log.txt"

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --dev-batch-size ${DEV_BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 100"
OPTS+=" --eval-interval 100"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --train-num ${TRAIN_NUM}"
OPTS+=" --dev-num 100"
OPTS+=" --seed ${SEED}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --few-data-num ${FEW_NUM}"
OPTS+=" --few-data-names ${FEW_DATA_NAMES}"

CMD="torchrun --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/train_t0.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
