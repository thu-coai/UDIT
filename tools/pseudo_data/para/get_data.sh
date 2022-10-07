WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/para/
PLAIN_TEXT=${WORKING_DIR}/plain_text/merge.txt

DATA_DIR=${WORKING_DIR}/data/

mkdir -p ${PSEUDO_DATA_TMP_DIR}

# construct pseudo para data
python3 ${WORKING_DIR}/get_pseudo_data/para/para.py \
    --input ${PLAIN_TEXT} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000

python3 ${WORKING_DIR}/get_pseudo_data/para/back_trans.py ${PSEUDO_DATA_TMP_DIR}/para.jsonl ${PSEUDO_DATA_TMP_DIR}/para_bt.jsonl

# construct pseudo para data (question)
python3 ${WORKING_DIR}/get_pseudo_data/para/question.py \
    --input ${PLAIN_TEXT} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000

python3 ${WORKING_DIR}/get_pseudo_data/para/back_trans.py ${PSEUDO_DATA_TMP_DIR}/question.jsonl ${PSEUDO_DATA_TMP_DIR}/question_bt.jsonl

# collect pseudo data for different instruction templates
python3 ${WORKING_DIR}/get_pseudo_data/para/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}