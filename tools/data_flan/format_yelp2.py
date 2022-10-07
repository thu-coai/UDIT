import json
import csv
import os

data_dir = "/home/yourname/data_en/yelp2"

data = []

options = ["negative", "positive"]

with open(os.path.join(data_dir, "train.csv")) as f:
    reader = csv.reader(f, delimiter=",", quotechar="\"")
    for row in reader:
        text = row[1].replace('\\""', '"').replace('\\n', ' ')
        data.append({
            "name": "yelp_polarity_reviews",
            "text": text,
            "options": options,
            "label": int(row[0]) - 1,
            "answer": options[int(row[0]) - 1]
        })


train_data = data[:-200]
valid_data = data[-200:]

with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
    for line in train_data:
        f.write(json.dumps(line) + "\n")


with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
    for line in valid_data:
        f.write(json.dumps(line) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
