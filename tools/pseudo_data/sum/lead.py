import argparse
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import torch
import random
import numpy as np

import json
from nltk.tokenize import sent_tokenize


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        pass

    def encode(self, line):
        line = line.split("<@x(x!>")
        
        if len(line[0]) < 100:
            line = line[1:]
        line = " ".join(line)
        sents = sent_tokenize(line)
        sents = sents[:-1]
        
        if len(sents) < 10:
            return None, len(line)

        summary = sents[:3]
        context = sents[3:]
        context = " ".join(context)
        summary = " ".join(summary)
        data = {
            "context": context,
            "summary": summary
        }

        return data, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="cc_news.txt", help='Path to input TXT')
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
    with open(os.path.join(args.output_dir, "lead.jsonl"), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    pool.close()

if __name__ == '__main__':
    main()