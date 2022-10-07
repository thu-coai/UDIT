import os
import json

data_dir = "/home/yourname/data_en/common_gen"

for split in ["train", "dev"]:
    with open(os.path.join(data_dir, "commongen.{}.jsonl".format(split))) as f:
        lines = f.readlines()
    data = []
    lines = [json.loads(line) for line in lines]
    for line in lines:
        concepts = ", ".join(line["concept_set"].split("#"))
        concepts_newline = "\n".join(line["concept_set"].split("#"))
        if split == "train":
            for tgt in line["scene"]:
                data.append({
                    "name": "common_gen",
                    "concepts": concepts,
                    "concepts_newline": concepts_newline,
                    "target": tgt,
                    "answers": []
                })
        else:
            data.append({
                "concepts": concepts,
                "concepts_newline": concepts_newline,
                "target": line["scene"][0],
                "answers": line["scene"]
            })
            
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
