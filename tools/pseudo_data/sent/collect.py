import json
import os
import sys

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

train_num = 9000
valid_num = 200


def yelp_polarity(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)
            
        new_sample = {
            "text": line["text"],
            "label": line["label"]
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "yelp_polarity", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def rotten_tomatoes(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)

        new_sample = {
            "text": line["text"],
            "label": line["label"]
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "rotten_tomatoes", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def imdb(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        line = json.loads(line)

        new_sample = {
            "text": line["text"],
            "label": line["label"]
        }
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "imdb", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


with open(os.path.join(input_data_dir, "imdb.jsonl")) as f:
    rotten_tomatoes(f)
    imdb(f)
    yelp_polarity(f)
