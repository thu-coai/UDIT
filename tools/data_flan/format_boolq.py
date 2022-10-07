import json
import os

data_dir = "/home/yourname/data_en/boolq/"

options = ['no', 'yes']

for split in ["train", "valid"]:
    data = []

    with open(os.path.join(data_dir, "{}_origin.jsonl".format(split))) as f:
        lines = f.readlines()

    for line in lines:
        line = json.loads(line)
        
        data.append({
            "name": "cb",
            "question": line["question"],
            "text": line["passage"],
            "options": options,
            "label": line["label"],
            "answer": options[int(line["label"])]
        })

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
