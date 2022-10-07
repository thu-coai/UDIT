import os
import json

data_dir = "/home/yourname/data_en/cola"

options = ["no", "yes"]

for split in ["train", "dev"]:
    data = []
    
    with open(os.path.join(data_dir, "CoLA", "{}.tsv".format(split))) as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip().split("\t")
        data.append({
            "name": "cola",
            "sentence": line[3],
            "label": int(line[1]),
            "options": options,
            "answer": options[int(line[1])]
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
