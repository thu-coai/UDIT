import json
import os

data_dir = "/home/yourname/data_en/cb/"

label_map = {
    "contradiction": "No",
    "neutral": "Maybe",
    "entailment": "Yes"
}

glm_label_map = {
    "contradiction": "false",
    "neutral": "neither",
    "entailment": "true"
}

options = ['Yes', 'No', 'Maybe']
glm_options = ['true', 'neither', 'false']


for split in ["train", "valid"]:
    with open(os.path.join(data_dir, "{}_origin.jsonl".format(split))) as f:
        lines = f.readlines()

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for line in lines:
            line = json.loads(line)
            line["name"] = "cb"
            line["options"] = options
            line["glm_options"] = glm_options
            line["answer"] = label_map[line["label"]]
            line["glm_answer"] = glm_label_map[line["label"]]
            
            f.write(json.dumps(line) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
