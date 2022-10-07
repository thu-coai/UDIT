import os
import json

data_dir = "/home/yourname/data_en/qqp"

options = ['no', 'yes']

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "QQP", "{}.tsv".format(split))) as f:
        lines = f.readlines()[1:]
        
    for line in lines:
        line = line.strip().split("\t")
        
        data.append({
            "name": "qqp",
            "question1": line[3],
            "question2": line[4],
            "options": options,
            "label": int(line[5]),
            "answer": options[int(line[5])]
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
