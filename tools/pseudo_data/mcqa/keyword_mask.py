import OpenHowNet
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
from collections import Counter

import json
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = TreebankWordDetokenizer()
        Encoder.hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)

    def encode(self, line):
        # end with <eod>
        all_data = []
        data = line.strip().replace("\u201d", "\"").replace("\u201c", "\"").replace(
                "\u2019", "\'").replace("\u2018", "\'").replace("\u2014", "-").replace("\u2013", "-").replace("``", "\"").replace("''", "\"")

        if "https" in data or "http" in data or ".com" in data:
            return None, 0

        data = data.split("<@x(x!>")
        data = [re.sub(r"\(.*\)", "", d) for d in data]
        
        if len(data) < 4:
            return None, 0
        
        for did, d in enumerate(data):
            if len(d) < 30 or len(d) > 300:
                continue

            d = word_tokenize(d)
            if random.random() < 0.5:
                cands = []
                for p, w in enumerate(d):
                    if re.match(r"[a-z]+", w):
                        if len(w) < 4:
                            continue
                        antonyms = []
                        for syn in wordnet.synsets(w):
                            for l in syn.lemmas():
                                if l.antonyms():
                                    antonyms.append(l.antonyms()[0].name())
                        if len(antonyms) > 0:
                            antonyms = list(set(antonyms))
                            ant = random.choice(antonyms)
                            cands.append((p, w, ant))
            else:
                count = Counter(d)
                tag = nltk.pos_tag(d)
                tag2w = {}
                for w, t in tag:
                    if t in tag2w:
                        tag2w[t].add(w)
                    else:
                        tag2w[t] = set()
                cands = []
                for p, (w, t) in enumerate(tag):
                    if re.match(r"[a-z]+", w):
                        if len(w) < 4:
                            continue
                        if t in ["NNP", "NNPS", "NN", "NNS"]:
                            if count[w] > 1:
                                if len(tag2w[t]) > 1:
                                    s = tag2w[t]
                                    s = list(s - {w})
                                    sim = [
                                        (o, self.hownet_dict_advanced.calculate_word_similarity(w, o)) for o in s]
                                    m = max(sim, key=lambda x: x[1])
                                    if m[1] > 0.7:
                                        cands.append((p, w, m[0]))
            
            if len(cands) == 0:
                continue        
             
            mask_p, mask_w, mask_ant = random.choice(cands)
            
            if mask_p == 0:
                new_d = ["_____"] + d[mask_p+1:]
            elif mask_p == len(d) - 1:
                new_d = d[:mask_p] + ["_____"]
            else:
                new_d = d[:mask_p] + ["_____"] + d[mask_p+1:]
            
            new_d = self.tokenizer.detokenize(new_d)

            if did == 0:
                context = data[did+1]
            elif did == len(data) - 1:
                context = data[did-1]
            else:
                if random.random() < 0.5:
                    context = data[did-1]
                else:
                    context = data[did+1]

            options = [mask_w, mask_ant]
            random.shuffle(options)
            label = options.index(mask_w)

            all_data.append({
                "context": context,
                "question": new_d,
                "options": options,
                "label": label
            })
            
        return all_data, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="merge.txt", help='Path to input TXT')
    parser.add_argument("--output_dir", default="pseudo_data_tmp/mcqa", type=str)
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

    for i, (samples, bytes_processed) in enumerate(encoded_docs, start=1):
        if samples is None:
            continue

        total_bytes_processed += bytes_processed

        all_samples.extend(samples)

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
    with open(os.path.join(args.output_dir, "keyword_mask.jsonl"), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    pool.close()

if __name__ == '__main__':
    main()