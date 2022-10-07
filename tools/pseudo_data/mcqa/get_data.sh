WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/mcqa/
PLAIN_TEXT=${WORKING_DIR}/plain_text/merge.txt
PLAIN_TEXT_SHUF1=${WORKING_DIR}/plain_text/merge_shuf_1.txt
PLAIN_TEXT_SHUF2=${WORKING_DIR}/plain_text/merge_shuf_2.txt
DATA_DIR=${WORKING_DIR}/data/

mkdir -p ${PSEUDO_DATA_TMP_DIR}

# construct pseudo mcqa data with keyword mask
python3 ${WORKING_DIR}/get_pseudo_data/mcqa/keyword_mask.py \
    --input ${PLAIN_TEXT} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000

# construct pseudo mcqa data with natual question
python3 ${WORKING_DIR}/get_pseudo_data/mcqa/natural_question.py \
    --input ${PLAIN_TEXT} \
    --input_neg1 ${PLAIN_TEXT_SHUF1} \
    --input_neg2 ${PLAIN_TEXT_SHUF2} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000 \
    --option_num 4

python3 ${WORKING_DIR}/get_pseudo_data/mcqa/natural_question.py \
    --input ${PLAIN_TEXT} \
    --input_neg1 ${PLAIN_TEXT_SHUF1} \
    --input_neg2 ${PLAIN_TEXT_SHUF2} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000 \
    --option_num 3

# collect pseudo data for different instruction templates
python3 ${WORKING_DIR}/get_pseudo_data/mcqa/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}