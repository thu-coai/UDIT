import json
import os
import random
import sys

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]


def agnews(all_data):
    ag_news = {
        "sport": all_data["sport"][:1083],
        "politics": all_data["politics"][:1083],
        "business": all_data["business"][:1083],
        "technology": all_data["technology"][:1083]
    }

    label_list = ["politics", "sport", "business", "technology"]
    print(label_list)
    ag_news_new = []
    for k in ag_news:
        for x in ag_news[k]:
            ag_news_new.append({
                "text": x["text"],
                "label": label_list.index(x["topic"])
            })

    random.shuffle(ag_news_new)
    valid_set = ag_news_new[:200]
    train_set = ag_news_new[200:]


    pseudo_dir = os.path.join(output_data_dir, "ag_news", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")


def dbpedia(all_data):
    ag_news_key = [
        "sport",
        "politics",
        "business",
        "technology"
    ]
    dbpedia = {}
    for k in all_data:
        if k != "technology":
            if k in ag_news_key:
                dbpedia[k] = all_data[k][1083:2183]
            else:
                dbpedia[k] = all_data[k][:1100]

    label_list = list(dbpedia.keys())
    dbpedia_new = []
    for k in dbpedia:
        for x in dbpedia[k]:
            paras = x["text"].strip().split("\n")
            title = paras[0]
            content = "\n".join(paras)
            dbpedia_new.append({
                "title": title,
                "content": content,
                "label": label_list.index(x["topic"])
            })

    random.shuffle(dbpedia_new)
    valid_set = dbpedia_new[:200]
    train_set = dbpedia_new[200:]

    pseudo_dir = os.path.join(output_data_dir, "dbpedia_14", "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    for split_set, split in [(train_set, "train"), (valid_set, "validation")]:
        with open(os.path.join(pseudo_dir, "{}.jsonl".format(split)), "w") as f:
            for d in split_set:
                f.write(json.dumps(d) + "\n")
                
                
with open(os.path.join(input_data_dir, "cc_news.json"), "r") as f:
    all_data = json.load(f)

random.seed(20)
for k in all_data:
    random.shuffle(all_data[k])

agnews(all_data)
dbpedia(all_data)
