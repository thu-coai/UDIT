import os
import json

data_dir = "/home/yourname/data_en/coqa/"

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "coqa-{}-v1.0.json".format(split))) as f:
        lines = json.load(f)
    for line in lines["data"]:
        questions_num = len(line["questions"])
        item_str = [str(i + 1) for i in range(questions_num)]
        questions = " ".join(i_str + ". " + q["input_text"] for i_str, q in zip(item_str, line["questions"]))
        answers = " ".join(i_str + ". " + an["input_text"] for i_str, an in zip(item_str, line["answers"]))
        
        data.append({
            "name": "coqa",
            "text": line["story"],
            "numbered_questions": questions,
            "numbered_answers": answers,
            "answer": answers
        })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
