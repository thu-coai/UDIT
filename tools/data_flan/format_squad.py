import os
import json

data_dir = "/home/yourname/data_en/squad"


for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "1.1", "{}-v{}.json".format(split, "1.1"))) as f:
        lines = json.load(f)
    
    for pas in lines["data"]:
        for para in pas["paragraphs"]:
            for qa in para["qas"]:
                data.append({
                    "name": "squad_v1",
                    "title": pas["title"],
                    "context": para["context"],
                    "question": qa["question"],
                    "answer": qa["answers"][0]["text"],
                    "answers": list(set([qa["answers"][i]["text"] for i in range(len(qa["answers"]))]))
                })

    with open(os.path.join(data_dir, "1.1", "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "1.1", "dev.jsonl"), os.path.join(data_dir, "1.1", "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "1.1", "valid.jsonl"), os.path.join(data_dir, "1.1", "test.jsonl")))


for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "2.0", "{}-v{}.json".format(split, "2.0"))) as f:
        lines = json.load(f)
    
    for pas in lines["data"]:
        for para in pas["paragraphs"]:
            for qa in para["qas"]:
                data.append({
                    "name": "squad_v2",
                    "title": pas["title"],
                    "context": para["context"],
                    "question": qa["question"],
                    "answer": "unanswerable" if qa["is_impossible"] else qa["answers"][0]["text"],
                    "answers": ["unanswerable"] if qa["is_impossible"] else list(set([qa["answers"][i]["text"] for i in range(len(qa["answers"]))]))
                })
        
    with open(os.path.join(data_dir, "2.0", "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "2.0", "dev.jsonl"), os.path.join(data_dir, "2.0", "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "2.0, ""valid.jsonl"), os.path.join(data_dir, "2.0, ""test.jsonl")))
