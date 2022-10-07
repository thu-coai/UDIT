import json
import os

data_dir = "/home/yourname/data_en/sst2"

options = ['negative', 'positive']

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "SST-2", "{}.tsv".format(split))) as f:
        lines = f.readlines()[1:]
    lines = [line.split("\t") for line in lines]
    for line in lines:
        data.append({
            "name": "sst2",
            "sentence": line[0],
            "label": int(line[1]),
            "options": options,
            "answer": options[int(line[1])]
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
