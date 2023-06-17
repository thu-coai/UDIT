# Learning Instructions with Unlabeled Data for Zero-Shot Cross-Task Generalization

Code and datasets for our paper "Learning Instructions with Unlabeled Data for Zero-Shot Cross-Task Generalization"

## 1 Environment

The code requires the CUDA10.2 toolkit. 

##### Install basic dependencies

```bash
pip install -r requirements.txt
```

##### Install apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
##### Fix DeepSpeed

Since there exist some **bugs** in DeepSpeed, you need to make some little modifications to this package. Specifically, you need to modify two lines of code in `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py`. We provide the modified `tools/ds_fix/stage1.py` and `tools/ds_fix/engine.py` in our repo. You can simply replace `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` with `stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py` with `engine.py` that we provided. 


## 2 Dataset

### 2.1 Labeled Data
All our datasets can be downloaded from the [HuggingFace Dataset](https://huggingface.co/datasets). You can download the original data and preprocess them using our scripts. For example, for the `Adversarial QA` Dataset, you can put the `json` files in `"/home/yourname/data_hf/adversarial_qa/`. Then, you can comment out the command for other datasets in `tools/data_t0/get_all_data.py` and run:
```bash
python3 tools/data_t0/get_all_data.py
```
This script will use tools/data_t0/adversarial_qa.py to process the data to .jsonl files. For other datasets, you can refer to the corresponding files for the approperate paths for the original data.

### 2.2 Pseudo Data
We also download the unlabeled plain texts from the [HuggingFace Dataset](https://huggingface.co/datasets). The scripts to construct pseudo data can be found in `tools/pseudo_data/`. You can run the `get_data.sh` script under the corresponding directory. Take MCQA as an example:

```bash
bash tools/pseudo_data/mcqa/get_data.sh

### 2.3 Evaluation Data
Our evaluation data can be download from this [link](https://huggingface.co/datasets/t1101675/UDIT_data).
```

## 3 Base Models

The original base model is obtained from HuggingFace. Before running the code, please use the transforming scripts to transfer the original pytorch_model.bin model checkpoints to fit in our DeepSpeed + Megatron framework:

```bash
mkdir -p checkpoints/t5-large-lm/t5-MP1

python3 tools/transform.py \
--hf_path ${PATH_TO_PYTORCH_MODLE_BIN}
--save_path "./checkpoints/t5-large-lm/t5-MP1"
--half
```

**Note that our base model is the [T5.1.1-lm100k](https://huggingface.co/liangtaiwan/t5-v1_1-lm100k-large)**

The pre-trained checkpoints can be download from this [link](https://huggingface.co/t1101675/UDIT/tree/main).


## 4 Run the Code

All scripts are in the directory `scripts`.

Before running the code, please first change the `WORKING_DIR` to the current directory of this repo. If you are runing multiple scripts on a single node, you need to make sure that the `MASTER_PORT` of each script is different. 

If the checkpoint is successfully loaded, the log printed to the stdout should contain messages like `successfully loaded /path-to-checkpoint/t5-MP4/mp_rank_01_model_states.pt`. Otherwise, `WARNING: could not find the metadata file /***/latest_checkpointed_iteration.txt will not load any checkpoints and will start from random` will display. Note that when you successfully load the model, you will see messages like `The following zero checkpoints paths are missing: ['/path-to-checkpoint/200000/zero_pp_rank_0_mp_rank_00_optim_states.pt',...` which mean optimizer states are not loaded. This **DOES NOT** affect the use of model inference and you can just ignore it.

### Vanilla-IT
```bash
bash scripts/it.sh
```

### UDIT (No Labeled Data)
```bash
bash scripts/udit_no_labeled.sh
```

### UDIT (Few Labeled Data)
```bash
bash scripts/udit_few_labeled.sh
```

### UDIT (Full Labeled Data)
```bash
bash scripts/udit_full_labeled.sh
```

### Zero-shot Cross-Task Evaluation
```bash
bash scripts/zs_fp16.sh ${PATH_TO_CHECKPOINT}
```

## 5 Citation
Please kindly cite our paper if you find this paper and the codes useful!
```
@inproceedings{udit,
    title = "Learning Instructions with Unlabeled Data for Zero-Shot Cross-Task Generalization",
    author = "Gu, Yuxian and Ke, Pei and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Proceedings of EMNLP",
    year = "2022",
}
```
