import json
import os

data_dir = "/home/yourname/data_en/stsb"

options = ["0", "1", "2", "3", "4", "5"]

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "STS-B", "{}.tsv".format(split))) as f:
        lines = f.readlines()[1:]
    
    for line in lines:
        line = line.strip().split("\t")
        data.append({
            "name": "stsb",
            "sentence1": line[7],
            "sentence2": line[8],
            "label": float(line[9]),
            "label_int": round(float(line[9])),
            "options": options,
            "answer": options[round(float(line[9]))]
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
