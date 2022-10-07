import json
import os
import random

random.seed(981217)

data_dir = "/home/yourname/data_en/mnli"

label_map = {
    "contradiction": "no",
    "entailment": "yes",
    "neutral": "it is not possible to tell"
}

options = ['yes', 'it is not possible to tell', 'no']

for split in ["train", "dev_matched", "dev_mismatched"]:
    data = []
    with open(os.path.join(data_dir, "MNLI", "original", "multinli_1.0_{}.jsonl".format(split))) as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    
    for line in lines:
        if line["gold_label"] == "-":
            continue
        data.append({
            "name": "mnli",
            "premise": line["sentence1"],
            "hypothesis": line["sentence2"],
            "label": line["gold_label"],
            "options": options,
            "answer": label_map[line["gold_label"]]
        })
        
    random.shuffle(data)
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("cat {} {} > {}".format(os.path.join(data_dir, "dev_matched.jsonl"), os.path.join(data_dir, "dev_mismatched.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
