WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/sum/
PLAIN_TEXT_1=${WORKING_DIR}/plain_text/cc_news.txt
PLAIN_TEXT_2=${WORKING_DIR}/plain_text/merge.txt

DATA_DIR=${WORKING_DIR}/data/

mkdir -p ${PSEUDO_DATA_TMP_DIR}

# construct pseudo sum data of Lead
python3 ${WORKING_DIR}/get_pseudo_data/sum/lead.py \
    --input ${PLAIN_TEXT_1} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000

# construct pseudo mcqa data of GSG
python3 ${WORKING_DIR}/get_pseudo_data/sum/gsg.py \
    --input ${PLAIN_TEXT_2} \
    --output_dir ${PSEUDO_DATA_TMP_DIR} \
    --max_sample_num 100000

# collect pseudo data for different instruction templates
python3 ${WORKING_DIR}/get_pseudo_data/sum/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}