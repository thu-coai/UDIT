import os
import re
import json

data_dir = "/home/yourname/data_en/hella_swag"

for split in ["train", "val"]:
    data = []
    with open(os.path.join(data_dir, "hellaswag_{}.jsonl.txt".format(split)), "r") as f:
        lines = [json.loads(line) for line in f.readlines()]
        
    for line in lines:
        context = re.sub(r'\[header\]\s', "", line["ctx"])
        context = re.sub(r'\[.*?\]\s', "\n", context)
        options = [re.sub(r'\[.*?\]\s', "", ending) for ending in line["endings"]]
        
        data.append({
            "name": "hellaswag",
            "context": context,
            "options": options,
            "label": line["label"],
            "answer": options[line["label"]]
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
        
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
