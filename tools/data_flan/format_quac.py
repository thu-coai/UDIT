import os
import json

data_dir = "/home/yourname/data_en/quac"

for split in ["train", "val"]:
    data = []
    with open(os.path.join(data_dir, "{}_v0.2.json".format(split))) as f:
        lines = json.load(f)
    lines = lines["data"]
    
    for line in lines:
        title = line["title"]
        background = line["background"]
        for para in line["paragraphs"]:
            context = para["context"]
            context = context.replace("CANNOTANSWER", "")
            for qa in para["qas"]:
                question = qa["question"]
                answer = "unanswerable" if qa["orig_answer"]["text"] == "CANNOTANSWER" else qa["orig_answer"]["text"]
                
        data.append({
            "name": "quac",
            "title": title,
            "background": background,
            "context": context,
            "question": question,
            "answer": answer
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
