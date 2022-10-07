import os
import json

data_dir = "/home/yourname/data_en/trec"

d = {
    "DESC": 'description',
    "ENTY": 'entity',
    "ABBR": 'abbreviation',
    "HUM": 'human',
    "NUM": 'numeric',
    "LOC": 'location'
}

options = ['description', 'entity', 'abbreviation', 'human', 'numeric', 'location']

data = []
with open(os.path.join(data_dir, "train_5500.label"), "rb") as f:
    lines = []
    for row in f:
        label, _, text = row.replace(b"\xf0", b" ").strip().decode().partition(" ")
        coarse_label, _, _ = label.partition(":")
        text = text.replace("``", "\"").replace("''", "\"").replace("`", "'")

        data.append({
            "name": "trec",
            "text": text,
            "label": coarse_label,
            "answer": d[coarse_label],
            "options": options
        })


train_data = data[:-200]
valid_data = data[-200:]

with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
    for d in train_data:
        f.write(json.dumps(d) + "\n")
        
with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
    for d in valid_data:
        f.write(json.dumps(d) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
