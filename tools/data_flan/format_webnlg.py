import os
import json

data_dir = "/home/yourname/data_en/web_nlg"

for split in ["train", "val"]:
    with open(os.path.join(data_dir, "webnlg_en_{}.json".format(split))) as f:
        lines = json.load(f)
        
    data = []
    for line in lines["values"]:
        input_str = "; ".join(line["input"])
        input_str = input_str.replace("_", " ").replace(" | ", ", ")
        
        if split == "train":
            for tgt in line["target"]:
                data.append({
                    "name": "web_nlg",
                    "input_string": input_str,
                    "target": tgt,
                    "answers": []
                })
        else:
            data.append({
                "name": "web_nlg",
                "input_string": input_str,
                "target": line["target"][0],
                "answers": line["target"]
            })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
