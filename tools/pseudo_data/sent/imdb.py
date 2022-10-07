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
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = TreebankWordDetokenizer()
        Encoder.analyzer = SentimentIntensityAnalyzer()

    def encode(self, line):
        # end with <eod>
        score = self.analyzer.polarity_scores(line)

        if score["neu"] < 0.78:
            if score["pos"] - score["neg"] > 0.05:
                label = 1
            else:
                label = 0

            sample = {
                "text": line,
                "label": label,
            }
        else:
            sample = None

        return sample, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="imdb.txt", help="Path to input TXT")
    parser.add_argument("--output_dir", default="pseudo_data_tmp/sent", type=str)
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes to launch")
    parser.add_argument("--log_interval", type=int, default=10000, help="Interval between progress updates")
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

    for i, (sample, bytes_processed) in enumerate(encoded_docs, start=1):
        if sample is None:
            continue
        total_bytes_processed += bytes_processed

        all_samples.append(sample)
        
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
    with open(os.path.join(args.output_dir, "imdb.jsonl"), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    pool.close()

if __name__ == '__main__':
    main()