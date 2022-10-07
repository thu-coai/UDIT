import json
import os

data_dir = "/home/yourname/data_en/anli/"

label_map = {
    "c": "No",
    "n": "Maybe",
    "e": "Yes"
}

glm_label_map = {
    "c": "false",
    "n": "neither",
    "e": "true"
}

options = ['Yes', 'No', 'Maybe']
glm_options = ['true', 'neither', 'false']

for r in [1, 2, 3]:
    for split in ["train", "dev"]:
        with open(os.path.join(data_dir, "anli_r{}".format(r), "R{}".format(r), "{}.jsonl".format(split))) as f:
            lines = f.readlines()
        
        lines = [json.loads(line) for line in lines]

        with open(os.path.join(data_dir, "anli_r{}".format(r), "{}.jsonl".format(split)), "w") as f:
            for line in lines:
                f.write(json.dumps({
                    "name": "anli",
                    "context": line["context"],
                    "hypothesis": line["hypothesis"],
                    "label": line["label"],
                    "options": options,
                    "glm_options": glm_options,
                    "answer": label_map[line["label"]],
                    "glm_answer": glm_label_map[line["label"]]
                }) + "\n")

    os.system("mv {} {}".format(os.path.join(data_dir, "anli_r{}".format(r), "dev.jsonl"), os.path.join(data_dir, "anli_r{}".format(r), "valid.jsonl")))
    os.system("cp {} {}".format(os.path.join(data_dir, "anli_r{}".format(r), "valid.jsonl"), os.path.join(data_dir, "anli_r{}".format(r), "test.jsonl")))
