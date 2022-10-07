import json
import os
import csv

data_dir = "/home/yourname/data_en/wnli"

options = ['no', 'yes']

for split in ["train", "dev"]:
    all_data = []
    with open(os.path.join(data_dir, "WNLI", "{}.tsv".format(split))) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            all_data.append(row[1:])
    
    all_data = all_data[1:]
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for line in all_data:
            f.write(json.dumps({
                "name": "wnli",
                "sentence1": line[0],
                "sentence2": line[1],
                "options": options,
                "label": int(line[2]),
                "answer": options[int(line[2])]
            }) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
