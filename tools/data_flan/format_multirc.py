import os
import json
import random

random.seed(1217)

data_dir = "/home/yourname/data_en/multirc"

options = ['no', 'yes']
glm_options = ['False', 'True']

for split in ["train", "val"]:
    data = []
    with open(os.path.join(data_dir, "MultiRC", "{}.jsonl".format(split))) as f:
        lines = f.readlines()
        lines = [json.loads(line.strip()) for line in lines]

    for line in lines:
        line = line["passage"]
        for q in line["questions"]:
            for a in q["answers"]:
                data.append({
                    "name": "multirc",
                    "paragraph": line["text"],
                    "question": q["question"],
                    "response": a["text"],
                    "options": options,
                    "glm_options": glm_options,
                    "answer": options[a["label"]],
                    "glm_answer": glm_options[a["label"]],
                    "label": a["label"]
                })

    random.shuffle(data)
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
