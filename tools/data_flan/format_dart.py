import os
import json
import re

data_dir = "/home/yourname/data_en/dart"

for split in ["train", "dev"]:
    with open(os.path.join(data_dir, "dart-v1.1.1-full-{}.json".format(split))) as f:
        lines = json.load(f)
    data = []
    
    for line in lines:
        tripleset = []
        for triple in line["tripleset"]:
            triple = [x for x in triple if re.fullmatch(r'\[(.*?)\]', x) is None]
            triple = ", ".join(triple)
        tripleset.append(triple)
        
        tripleset_newline = "\n".join(tripleset)
        tripleset = "; ".join(tripleset)
        
        if split == "train":
            for ann in line["annotations"]:
                data.append({
                    "name": "dart",
                    "tripleset": tripleset,
                    "tripleset_newline": tripleset_newline,
                    "target": ann["text"],
                    "answers": []
                })
        else:
            data.append({
                "name": "dart",
                "tripleset": tripleset,
                "tripleset_newline": tripleset_newline,
                "target": line["annotations"][0]["text"],
                "answers": [ann["text"] for ann in line["annotations"]]
            })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
