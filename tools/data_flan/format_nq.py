import os
import json
import re
import html

data_dir = "/home/yourname/data_en/nq"


for split in ["train", "dev"]:
    data = []
    did_lim = 5 if split == "dev" else 50
    for did in range(did_lim):
        with open(os.path.join(data_dir, "nq-{}-{:0>2}.jsonl".format(split, did))) as f:
            idx = 0
            for line in f:
                if idx % 1000 == 0:
                    print("nq-{}-{:0>2}.jsonl".format(split, did), idx)
                line = json.loads(line)
                html_bytes = line["document_html"].encode("utf-8")
                anses = []
                for anno in line["annotations"]:
                    for sans in anno["short_answers"]:
                        ans_bytes = html_bytes[sans["start_byte"] : sans["end_byte"]]
                        ans_bytes = ans_bytes.replace(b"\xc2\xa0", b" ")
                        text = ans_bytes.decode("utf-8")
                        # Remove HTML markup.
                        text = re.sub("<([^>]*)>", "", html.unescape(text))
                        anses.append(text)
                
                    if anno["yes_no_answer"] != "NONE":
                        anses.append(anno["yes_no_answer"])

                data.append({
                    "name": "nq",
                    "question": line["question_text"],
                    "answers": list(set(anses))
                })
                idx += 1

    with open(os.path.join(data_dir, "{}_full.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    data_squeeze = []
    for d in data:
        if len(d["answers"]) != 0:
            d["question"] = d["question"] + "?"
            d["answer"] = d["answers"][0]
            data_squeeze.append(d)
            
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
