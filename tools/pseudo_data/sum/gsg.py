import argparse
import multiprocessing
import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import torch
import random
import numpy as np
from t5.evaluation import metrics as t5_metrics

import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = TreebankWordDetokenizer()

    def encode(self, line):
        data = line.strip().replace("\u201d", "\"").replace("\u201c", "\"").replace(
                "\u2019", "\'").replace("\u2018", "\'").replace("\u2014", "-").replace("\u2013", "-").replace("``", "\"").replace("''", "\"")

        if "https" in data or "http" in data or ".com" in data:
            return None, len(line)

        data = data.split("<@x(x!>")
        data = [re.sub(r"\(.*\)", "", d) for d in data]
        data = " ".join(data)

        data = sent_tokenize(data)

        if len(data) < 3:
            return None, len(line)

        score = []
        for sid, sent in enumerate(data):
            if "?" in sent:
                score.append((sid, 0))
            else:
                rest = " ".join(data[:sid] + data[sid+1:])
                res = t5_metrics.rouge([sent], [rest], score_keys=["rouge1"])
                score.append((sid, res["rouge1"]))
        
        if len(score) == 0:
            return None, len(line)

        score = sorted(score, key=lambda x: x[1], reverse=True)

        sum_id, s = score[0]
        if s < 20:
            return None, len(line)

        data = {
            "context": " ".join(data[:sum_id] + data[sum_id+1:]),
            "summary": data[sum_id]
        }

        return data, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="merge.txt", help='Path to input TXT')
    parser.add_argument("--output_dir", default="pseudo_data_tmp/sum/", type=str)
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes to launch")
    parser.add_argument("--log_interval", type=int, default=1000, help="Interval between progress updates")
    parser.add_argument("--max_sample_num", type=int, default=-1)

    args = parser.parse_args()
    args.rank = 0

    return args


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    args = get_args()
    startup_start = time.time()

    set_random_seed(20)

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
    all_samples = []

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (samples, bytes_processed) in enumerate(encoded_docs, start=1):
        if samples is None:
            continue
        total_bytes_processed += bytes_processed

        all_samples.append(samples)

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print("Processed {} documents, ({} docs/s, {} MB/s). {} Samples".format(
                i, i/elapsed, mbs, len(all_samples)), file=sys.stderr)

        if args.max_sample_num > 0 and len(all_samples) > args.max_sample_num:
            break

    random.shuffle(all_samples)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "gsg.jsonl"), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    pool.close()

if __name__ == '__main__':
    main()