import json
import os
import sys

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

train_num = 20000
valid_num = 300


def qqp(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            "question1": line["sentence1"],
            "question2": line["sentence2"],
            "label": line["label"],
            "idx": idx
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "qqp", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def mrpc(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            **line,
            "idx": idx
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "mrpc", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def paws(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            **line,
            "idx": idx
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "qqp", "pseudo", "labeled_final")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


with open(os.path.join(input_data_dir, "para_bt.jsonl")) as f:
    mrpc(f)
    paws(f)

with open(os.path.join(input_data_dir, "question_bt.jsonl")) as f:
    qqp(f)
