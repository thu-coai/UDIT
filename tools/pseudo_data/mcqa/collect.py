import json
import os
import sys

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

train_num = 20000
valid_num = 100


def quartz(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        sample = json.loads(line)
        new_sample = {
            "id": idx,
            "para": sample["context"],
            "question": sample["question"],
            "choices": {"text": sample["options"], "label": ["A", "B"]},
            "answerKey": "A" if sample["label"] == 0 else "B"
        }

        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < train_num + valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "quartz", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def quarel(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        sample = json.loads(line)
        new_sample = {
            "id": idx,
            "question": sample["question"] + " (A) " + sample["options"][0] + " (B) " + sample["options"][1],
            "answer_index": sample["label"],
            "world_literals": {"world1": [sample["options"][0]], "world2": [sample["options"][1]]}
        }

        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < train_num + valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "quarel", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def social_i_qa(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        sample = json.loads(line)
        new_sample = {
            "id": idx,
            "context": sample["context"],
            "question": sample["question"],
            "answerA": sample["options"][0],
            "answerB": sample["options"][1],
            "answerC": sample["options"][2],
            "label": str(sample["label"] + 1)
        }

        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < train_num + valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "social_i_qa", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def cosmos_qa(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        sample = json.loads(line)
        new_sample = {
            "id": idx,
            "context": sample["context"],
            "question": sample["question"],
            "answer0": sample["options"][0],
            "answer1": sample["options"][1],
            "answer2": sample["options"][2],
            "answer3": sample["options"][3],
            "label": sample["label"]
        }
        
        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < train_num + valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "cosmos_qa", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def quail(f):
    train_set, valid_set = [], []
    idx = 0
    while True:
        line = next(f)
        sample = json.loads(line)
        new_sample = {
            "id": idx,
            "context": sample["context"],
            "question": sample["question"],
            "answers": sample["options"],
            "correct_answer_id": sample["label"]
        }

        idx += 1
        if len(train_set) < train_num:
            train_set.append(new_sample)
        elif len(valid_set) < train_num + valid_num:
            valid_set.append(new_sample)
        else:
            break

    pseudo_dir = os.path.join(output_data_dir, "quail", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


with open(os.path.join(input_data_dir, "natural_question_option4.jsonl")) as f:
    cosmos_qa(f)
    quail(f)

with open(os.path.join(input_data_dir, "natural_question_option3.jsonl")) as f:
    social_i_qa(f)

with open(os.path.join(input_data_dir, "keyword_mask.jsonl")) as f:
    quartz(f)
    quarel(f)
