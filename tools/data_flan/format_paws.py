import json
import os

data_dir = "/home/yourname/data_en/paws"

options = ['no', 'yes']

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "final", "{}.tsv".format(split))) as f:
        lines = f.readlines()[1:]
    lines = [line.split("\t") for line in lines]
    for line in lines:
        data.append({
            "name": "paws",
            "sentence1": line[1],
            "sentence2": line[2],
            "label": int(line[3]),
            "options": options,
            "answer": options[int(line[3])]
        })

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
