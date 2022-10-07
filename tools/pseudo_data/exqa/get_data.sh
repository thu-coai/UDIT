WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/exqa/
PLAIN_TEXT=${WORKING_DIR}/plain_text/wiki.txt
DATA_DIR=${WORKING_DIR}/data/

mkdir -p ${PSEUDO_DATA_TMP_DIR}

# spark-submit --master local[90] --driver-memory 200G ${WORKING_DIR}/get_pseudo_data/exqa/tokenize_and_ner_inputs.py --corpus=${PLAIN_TEXT}  --output ${PSEUDO_DATA_TMP_DIR}/sent-tok-rollup
# spark-submit --master local[90] --driver-memory 4G ${WORKING_DIR}/get_pseudo_data/exqa/write_sentence_level_es_index.py --corpus=${PSEUDO_DATA_TMP_DIR}/rollup --es-index uqa-es-index --output ${PSEUDO_DATA_TMP_DIR}/write-es
# spark-submit --master local[90] --driver-memory 300G ${WORKING_DIR}/get_pseudo_data/exqa/create_ds_synthetic_dataset.py --corpus=${PSEUDO_DATA_TMP_DIR}/rollup --output ${PSEUDO_DATA_TMP_DIR}/pseudo.jsonl --aux-qs=1 --aux-awc=1 --ulim-count=500000

python3 ${WORKING_DIR}/get_pseudo_data/exqa/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}