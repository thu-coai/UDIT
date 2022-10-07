import os
import json

data_dir = "/home/yourname/data_en/wic"

options = ['different meanings', 'the same meaning']

for split in ["train", "val"]:
    data = []
    
    with open(os.path.join(data_dir, "WiC", "{}.jsonl".format(split))) as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
    for line in lines:
        data.append({
            "name": "wic",
            'sentence1': line['sentence1'],
            'sentence2': line['sentence2'],
            'word': line["word"],
            'options': options,
            'answer': options[int(line["label"])],
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
