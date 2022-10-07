import os
import json
import csv

data_dir = "/home/yourname/data_en/cosmos_qa"


for split in ["train", "valid"]:
    data = []
    with open(os.path.join(data_dir, "{}.csv".format(split))) as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        i = 0        
        for item in reader:
            if i > 1:
                options = item[3:7]
                data.append({
                    "name": "cosmos_qa",
                    "context": item[1],
                    "question": item[2],
                    "options": options,
                    "answer": options[int(item[7])],
                    "label": int(item[7])
                })
            i += 1
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
