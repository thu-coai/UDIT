import json
import os
import random
from nltk.tokenize import sent_tokenize
import sys

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

train_num = 12000
valid_num = 200


def adqa(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            "title": line["meta"]["context"]["article_title"],
            "context": line["context"],
            "question": line["question"],
            "answers": {
                "text": [x["text"] for x in line["answers"]],
                "answer_start": [x["answer_start"] for x in line["answers"]]
            },
            "id": line["qid"]
        }
        idx += 1
        if len(train_set) < train_num:
            new_sample["metadata"] = {
                "split": "train"
            }
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            new_sample["metadata"] = {
                "split": "validation"
            }
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "adversarial_qa", "pseudo", "adversarialQA")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def quoref(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            "title": line["meta"]["context"]["article_title"],
            "context": line["context"],
            "question": line["question"],
            "answers": {
                "text": [x["text"] for x in line["answers"]],
                "answer_start": [x["answer_start"] for x in line["answers"]]
            },
            "id": line["qid"]
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "quoref", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")
                
                
def ropes(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        
        sents = sent_tokenize(line["context"])
        background = sents[:-1]
        situation = sents[-1]
        if len(background) == 0:
            continue
        new_sample = {
            "background": background,
            "situation": situation,
            "question": line["question"],
            "answers": {
                "text": [x["text"] for x in line["answers"]],
            },
            "id": line["qid"]
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "ropes", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")

random.seed(42)

with open(os.path.join(input_data_dir, "pseudo.jsonl")) as f:
    adqa(f)
    quoref(f)
    ropes(f)
