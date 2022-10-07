import json
import os

data_dir = "/home/yourname/data_en/rte"

label_map = {
    "not_entailment": "no",
    "entailment": "yes"
}

glm_label_map = {
    "not_entailment": "false",
    "entailment": "true"
}

options = ["yes", "no"]
glm_options = ["true", "false"]


for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "RTE", "{}.tsv".format(split))) as f:
        lines = f.readlines()[1:]
    lines = [line.strip().split("\t") for line in lines]
    
    for line in lines:
        data.append({
            "name" : "rte",
            "premise": line[1],
            "hypothesis": line[2],
            "options" : options,
            "glm_options" : glm_options,
            "answer" : label_map[line[3]],
            "glm_answer" : glm_label_map[line[3]],
        })

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
