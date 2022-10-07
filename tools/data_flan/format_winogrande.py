import os
import json

data_dir = "/home/yourname/data_en/winogrande"

for split in ["train_xl", "dev"]:
    data = []
    with open(os.path.join(data_dir, "winogrande_1.1", "{}.jsonl".format(split))) as f:
        lines = [json.loads(line) for line in f.readlines()]
        
    for line in lines:
        context, next_sentence = line["sentence"].split("_")
        options = [line["option1"] + next_sentence, line["option2"] + next_sentence]
        data.append(
            {
                "name": "winogrande",
                "context": context,
                "options": options,
                "label": int(line["answer"]) - 1,
                "answer": options[int(line["answer"]) - 1]
            }
        )
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("mv {} {}".format(os.path.join(data_dir, "train_xl.jsonl"), os.path.join(data_dir, "train.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
