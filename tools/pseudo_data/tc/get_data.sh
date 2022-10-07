WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/tc/
PLAIN_TEXT=${WORKING_DIR}/plain_text/cc_news_url.jsonl
DATA_DIR=${WORKING_DIR}/data/
TOKENIZER_PATH=${WORKING_DIR}/vocab_en/spiece.model

mkdir -p ${PSEUDO_DATA_TMP_DIR}

# construct pseudo mcqa data with keyword mask
python3 ${WORKING_DIR}/get_pseudo_data/tc/cc_news.py \
    --input ${PLAIN_TEXT} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000 \
    --tokenizer_path ${TOKENIZER_PATH}

# collect pseudo data for different instruction templates
python3 ${WORKING_DIR}/get_pseudo_data/tc/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}