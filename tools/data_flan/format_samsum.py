import json
import os
import re

data_dir = "/home/yourname/data_en/samsum"

for split in ["train", "val"]:
    data = []
    with open(os.path.join(data_dir, "corpus", "{}.json".format(split))) as f:
        lines = json.load(f)
    
    data = []
    for line in lines:
        dialogue = re.sub(r"\r\n", "\n", line["dialogue"])
        dialogue = re.sub(r"<.*>", " ", dialogue)
        data.append({
            "name": "samsum",
            "dialogue": dialogue,
            "summary": line["summary"],
            "answer": line["summary"]
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
