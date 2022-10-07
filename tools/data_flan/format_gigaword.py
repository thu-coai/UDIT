import os
import json
import re

data_dir = "/home/yourname/data_en/gigaword/"

for split in ["train", "dev"]:
    data = []
    all_src = set()
    with open(os.path.join(data_dir, "org_data", "{}.src.txt".format(split))) as f:
        src_lines = f.readlines()
        
    with open(os.path.join(data_dir, "org_data", "{}.tgt.txt".format(split))) as f:
        tgt_lines = f.readlines()
        
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        tmp_line = src_line + tgt_line
        tmp_line = tmp_line.replace("\n", "")
        if re.fullmatch(r".*UNK.*", tmp_line) is not None or re.fullmatch(r".*#.*", tmp_line) is not None:
            continue
        if src_line in all_src:
            continue
        all_src.add(src_line)
        data.append({
            "text": src_line.strip(),
            "answer": tgt_line.strip()
        })
        
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "dev.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
