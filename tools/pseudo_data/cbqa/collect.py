import json
import os
import random
import sys

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

train_num = 12000
valid_num = 200

def wiki_qa(lines, all_answers):
    train_set, valid_set = [], []
    idx = 0
    for line in lines:
        true_answer = line["answers"][0]["text"]

        if random.random() < 0.5:
            answer = true_answer
            label = 1
        else:
            answer = true_answer
            while answer == true_answer:
                answer = random.choice(all_answers)
            label = 0

        new_sample = {
            "question_id": idx,
            "question": line["question"],
            "document_title": line["meta"]["context"]["article_title"],
            "answer": answer,
            "label": label
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "wiki_qa", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


with open(os.path.join(input_data_dir, "pseudo.jsonl")) as f:
    lines = f.readlines()

lines = [json.loads(line) for line in lines]
all_answers = [[x["text"] for x in line["answers"]] for line in lines]
all_answers = [x for y in all_answers for x in y]

random.seed(42)
random.shuffle(lines)

wiki_qa(lines, all_answers)
