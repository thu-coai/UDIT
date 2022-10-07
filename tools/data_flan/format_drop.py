import os
import json

data_dir = "/home/yourname/data_en/drop"

for split in ["train", "dev"]:
    data = []
    with open(os.path.join(data_dir, "drop_dataset", "drop_dataset_{}.json".format(split))) as f:
        lines = json.load(f)
        
    data = []
    for k, v in lines.items():
        passage = v["passage"]
        for qa in v["qa_pairs"]:
            question = qa["question"]
            all_answer_cands = []
            answer_all = qa["answer"]
            number = answer_all.get("number", "")
            date = answer_all.get("data", {})
            day = date.get("day", "")
            month = date.get("month", "")
            year = date.get("year", "")
            spans = " ".join(answer_all.get("spans", []))
            answer = " ".join([number, day, month, year, spans]).strip()
            if len(answer) == 0:
                continue
            all_answer_cands.append(answer)
            if split == "dev":
                for val_answer_all in qa["validated_answers"]:
                    number = val_answer_all.get("number", "")
                    date = val_answer_all.get("data", {})
                    day = date.get("day", "")
                    month = date.get("month", "")
                    year = date.get("year", "")
                    spans = " ".join(val_answer_all.get("spans", []))
                    val_answer = " ".join([number, day, month, year, spans]).strip()
                    if len(val_answer) == 0:
                        continue
                    all_answer_cands.append(val_answer)
                all_answer_cands = list(set(all_answer_cands))
                
            data.append({
                "name": "drop",
                "context": passage,
                "question": question,
                "answer": answer,
                "answers": all_answer_cands
            })
    
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
