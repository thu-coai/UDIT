WORKING_DIR=/home/yourname/UDIT

PSEUDO_DATA_TMP_DIR=${WORKING_DIR}/pseudo_data_tmp/cbqa/
DATA_DIR=${WORKING_DIR}/data/

mkdir -p ${PSEUDO_DATA_TMP_DIR}

cp ${WORKING_DIR}/pseudo_data_tmp/exqa/pseudo.jsonl ${PSEUDO_DATA_TMP_DIR}/pseudo.jsonl

python3 ${WORKING_DIR}/get_pseudo_data/cbqa/collect.py ${PSEUDO_DATA_TMP_DIR} ${DATA_DIR}