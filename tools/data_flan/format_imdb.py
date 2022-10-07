import json
import csv
import os

data_dir = "/home/yourname/data_en/imdb"

options = ["negative", "positive"],

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "{}.csv".format(split))) as f:
        reader = csv.reader(f, delimiter="\t", quotechar="\"")
        for row in reader:
            sent = row[0]
            sent = sent.replace("<br />", " ")
            sent = sent[1:-1]
            sent = sent.replace("\"\"", "\"")
            data.append({
                "name": "imdb",
                "text": sent,
                "options": options,
                "label": row[1],
                "answer": row[1]
            })

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
