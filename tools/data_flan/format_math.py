import json
import os
import random

random.seed(981217)

data_dir = "/home/yourname/data_en/math"

data = []
for type in ["easy", "medium", "hard"]:
    with open(os.path.join(data_dir, "mathematics_dataset-v1.0", "train-{}".format(type), "algebra__linear_1d.txt")) as f:
        lines = f.readlines()
        
    for i in range(0, len(lines), 2):
        question = lines[i].strip()
        answer = lines[i+1].strip()
        data.append({
            "name": "math",
            "question": question,
            "answer": answer
        })

random.shuffle(data)
train_data = data[:-200]
valid_data = data[-200:]

with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
    for d in train_data:
        f.write(json.dumps(d) + "\n")
        
with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
    for d in valid_data:
        f.write(json.dumps(d) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
