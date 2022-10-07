import os
import json

data_dir = "/home/yourname/data_en/mrpc"

options = ['no', 'yes']

with open(os.path.join(data_dir, "msr_paraphrase_train.txt")) as f:
    lines = f.readlines()[1:]
    
    
with open(os.path.join(data_dir, "mrpc_dev_ids.tsv")) as f:
    dev_idx_lines = f.readlines()
     
all_data = {}

for line in lines:
    line = line.strip().split("\t")
    d = {
        "name": "mrpc",
        "sentence1": line[3],
        "sentence2": line[4],
        "options": options,
        "label": int(line[0]),
        "answer": options[int(line[0])]
    }
    all_data[(int(line[1]), int(line[2]))] = d
    
train_data = []
valid_data = []    

all_dev_idx = []
for dev_idx_line in dev_idx_lines:
    idx = dev_idx_line.strip().split("\t")
    tmp_idx = (int(idx[0]), int(idx[1]))
    all_dev_idx.append(tmp_idx)

all_dev_idx = set(all_dev_idx)

for k, v in all_data.items():
    if k in all_dev_idx:
        valid_data.append(v)
    else:
        train_data.append(v)
        
with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
    for d in train_data:
        f.write(json.dumps(d) + "\n")
        
with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
    for d in valid_data:
        f.write(json.dumps(d) + "\n")
        
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
