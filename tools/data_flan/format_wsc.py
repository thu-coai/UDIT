import os
import json

data_dir = "/home/yourname/data_en/wsc"

options = ['no', 'yes']

for split in ["train", "valid"]:
    data = []
    with open(os.path.join(data_dir, "WSC", "{}_origin.jsonl".format(split))) as f:
        lines = [json.loads(line) for line in f.readlines()]
        
    for line in lines:
        data.append({
            "name": "wsc",
            "context": line["text"],
            "text1": line["target"]["span1_text"],
            "text2": line["target"]["span2_text"],
            "options": options,
            "label": line["label"],
            "answer": options[int(line["label"])]
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
