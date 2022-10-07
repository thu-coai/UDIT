import os
import json
import csv

data_dir = "/home/yourname/data_en/agnews"

options = ['World', 'Sports', 'Business', 'Science/Tech']

for split in ["train", "test"]:
    data = []
    with open(os.path.join(data_dir, "{}.csv".format(split))) as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        
        for line in reader:
            data.append({
                "name": "ag_news_subset",
                "title": line[1],
                "text": line[2],
                "options": options,
                "label": int(line[0]),
                "answer": options[int(line[0]) - 1]
            })
            
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "test.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
