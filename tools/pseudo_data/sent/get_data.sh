WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/sent/
PLAIN_TEXT=${WORKING_DIR}/plain_text/imdb.txt
DATA_DIR=${WORKING_DIR}/data/

mkdir -p ${PSEUDO_DATA_TMP_DIR}

# construct pseudo mcqa data with keyword mask
python3 ${WORKING_DIR}/get_pseudo_data/sent/imdb.py \
    --input ${PLAIN_TEXT} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000

# collect pseudo data for different instruction templates
python3 ${WORKING_DIR}/get_pseudo_data/sent/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}