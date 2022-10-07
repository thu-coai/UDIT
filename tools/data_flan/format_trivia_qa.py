import os
import json

data_dir = "/home/yourname/data_en/trivia_qa"

for split in ["train", "dev"]:
    data = []
    for domain in ["web", "wikipedia"]:
        with open(os.path.join(data_dir, "qa", "{}-{}.json".format(domain, split))) as f:
            lines = json.load(f)

        lines = lines["Data"]
        for line in lines:
            data.append({
                "name": "trivia_qa",
                "question": line["Question"],
                "answer": line["Answer"]["NormalizedValue"],
                "answers": line["Answer"]["NormalizedAliases"]
            })

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
