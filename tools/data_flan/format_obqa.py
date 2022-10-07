import os
import json

data_dir = "/home/yourname/data_en/obqa/"

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "OpenBookQA-V1-Sep2018/Data/Additional", "{}_complete.jsonl".format(split))) as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    data = []
    for line in lines:
        options = [op["text"] for op in line["question"]["choices"]]
        data.append({
            "name": "obqa",
            "question": line["question"]["stem"],
            "options": options,
            "fact": line["fact1"],
            "answer": options[ord(line["answerKey"]) - ord('A')]
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
