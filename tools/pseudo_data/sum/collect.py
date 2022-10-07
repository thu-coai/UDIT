import json
import os
import sys
from nltk import sent_tokenize

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

train_num = 30000
valid_num = 1000


def cnn_dm(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            "article": line["context"],
            "highlights": line["summary"],
            "idx": idx
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break
    
    pseudo_dir = os.path.join(output_data_dir, "cnn_dailymail", "pseudo", "3.0.0")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def xsum(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        sents = sent_tokenize(line["context"])
        if len(sents) < 5:
            continue
        new_sample = {
            "document": line["context"],
            "summary": line["summary"],
            "idx": idx
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "xsum", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def gigaword(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
        new_sample = {
            "document": line["context"],
            "summary": line["summary"],
            "idx": idx
        }
        idx += 1
        if len(train_set) < 11000:
            train_set.append(new_sample)
        elif len(valid_set) < 200:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "gigaword", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


with open(os.path.join(input_data_dir, "gsg.jsonl")) as f:
    xsum(f)

with open(os.path.join(input_data_dir, "lead.jsonl")) as f:
    cnn_dm(f)
    gigaword(f)
