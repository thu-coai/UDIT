import os
import json
import csv
import random

random.seed(981217)

data_dir = "/home/yourname/data_en/e2e"

for split in ["train", "devel"]:
    data = []
    with open(os.path.join(data_dir, "{}-fixed.no-ol.csv".format(split))) as f:
        reader = csv.DictReader(f)
        tmp_data = {}
        for line in reader:
            meaning_representation = line["mr"].replace("[", " = ").replace("]", "")
            if meaning_representation in tmp_data:
                tmp_data[meaning_representation]["answers"].append(line["ref"])
            else:
                tmp_data[meaning_representation] = {
                    "name": "e2e_nlg",
                    "meaning_representation": meaning_representation,
                    "target": line["ref"],
                    "answers": [line["ref"]]
                }
        
        for v in tmp_data.values():
            data.append(v)

    random.shuffle(data)
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")    

os.system("mv {} {}".format(os.path.join(data_dir, "devel.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
