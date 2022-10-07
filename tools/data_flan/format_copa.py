import os
import json

data_dir = "/home/yourname/data_en/copa"

for split in ["train", "val"]:
    data = []
    with open(os.path.join(data_dir, "COPA", "{}.jsonl".format(split))) as f:
        lines = [json.loads(line) for line in f.readlines()]
        
    for line in lines:
        data.append({
            "name": "copa",
            "premise": line["premise"],
            "question": line["question"],
            "options": [line["choice1"], line["choice2"]],
            "label": line["label"],
            "answer": line["choice1"] if line["label"] == 0 else line["choice2"]
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
