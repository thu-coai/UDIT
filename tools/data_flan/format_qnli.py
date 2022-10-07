import json
import os

data_dir = "/home/yourname/data_en/qnli"

label_map = {
    "not_entailment": "no",
    "entailment": "yes"
}

options = ["yes", "no"]

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "QNLI", "{}.tsv".format(split))) as f:
        lines = f.readlines()[1:]
    lines = [line.strip().split("\t") for line in lines]
    
    for line in lines:
        data.append({
            "name": "qnli",
            "question": line[1],
            "sentence": line[2],
            "options": options,
            "label": line[3],
            "answer": label_map[line[3]]
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
