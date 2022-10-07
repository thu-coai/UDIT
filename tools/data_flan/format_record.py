import os
import json

data_dir = "/home/yourname/data_en/record"

for split in ["train", "val"]:
    with open(os.path.join(data_dir, "ReCoRD", "{}.jsonl".format(split))) as f:
        lines = f.readlines()
        
    data = []
    for line in lines:
        line = json.loads(line)
        pas = line["passage"]
        options = list(set([pas["text"][e["start"]:e["end"]+1] for e in pas["entities"]]))
        for qa in line["qas"]:
            query_left, query_right = qa["query"].split("@placeholder")
            long_options = [op + query_right for op in options]
            answers = list(set([ans["text"] for ans in qa["answers"]]))
            long_answers = [ans + query_right for ans in answers]
            data.append({
                "name": "record",
                "passage": pas["text"].replace("\n@highlight", ""),
                "query": query_left,
                "options_origin": options,
                "options": long_options,
                "labels": answers,
                "answer_origin": answers,
                "answers": long_answers,
                "answer": long_answers[0]
            })
            
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
