import os
import json
from tqdm import tqdm

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)

data_dir = "/home/yourname/data_en/xsum"

with open(os.path.join(data_dir, "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json")) as f:
    all_idxs = json.load(f)
    
for split in ["train", "validation"]:
    idxs = all_idxs[split]
    data = []
    for idx in tqdm(idxs):
        with open(os.path.join(data_dir, "bbc-summary-data", "{}.summary".format(idx))) as f:
            d = f.readlines()
            d = " ".join([x for x in d if x not in _REMOVE_LINES])
            d = d.split("[SN]")
        
        data.append({
            "name": "xsum",
            "text": d[8].strip(),
            "answer": d[6].strip()
        })

    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
os.system("mv {} {}".format(os.path.join(data_dir, "validation.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
