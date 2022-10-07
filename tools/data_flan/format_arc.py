import os
import json

data_dir = "/home/yourname/data_en/arc/"

for type in ["Challenge", "Easy"]:
    for split in ["Train", "Dev"]:
        data = []
        with open(os.path.join(data_dir, "ARC-V1-Feb2018-2/ARC-{}".format(type), "ARC-{}-{}.jsonl".format(type, split))) as f:
            lines = [json.loads(line) for line in f.readlines()]
        
        data = []
        for line in lines:
            options = [op["text"] for op in line["question"]["choices"]]
            if line["answerKey"] in ["A", "B", "C", "D", "E"]:
                answer = options[ord(line["answerKey"]) - ord('A')]
            else:
                answer = options[ord(line["answerKey"]) - ord('1')]

            data.append({
                "name": "arc_{}".format(type.lower()),
                "question": line["question"]["stem"],
                "options": options,
                "answer": answer
            })
            
        with open(os.path.join(data_dir, type.lower(), "{}.jsonl".format(split)), "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")

    os.makedirs(os.path.join(data_dir, type.lower()), exist_ok=True)

    os.system("mv {} {}".format(os.path.join(data_dir, type.lower(), "Train.jsonl"), os.path.join(data_dir, type.lower(), "train.jsonl")))
    os.system("mv {} {}".format(os.path.join(data_dir, type.lower(), "Dev.jsonl"), os.path.join(data_dir, type.lower(), "valid.jsonl")))
    os.system("cp {} {}".format(os.path.join(data_dir, type.lower(), "valid.jsonl"), os.path.join(data_dir, type.lower(), "test.jsonl")))
