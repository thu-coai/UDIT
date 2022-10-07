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
import string
import copy

import json
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = TreebankWordDetokenizer()

    def encode(self, line):
        # end with <eod>
        all_data = []
        data = line.strip().replace("\u201d", "\"").replace("\u201c", "\"").replace(
                "\u2019", "\'").replace("\u2018", "\'").replace("\u2014", "-").replace("\u2013", "-").replace("``", "\"").replace("''", "\"")

        if "https" in data or "http" in data or ".com" in data:
            return None, 0

        data = data.split("<@x(x!>")
        data = [re.sub(r"\(.*\)", "", d) for d in data]
        
        data = " ".join(data)
        data = sent_tokenize(data)
        
        if len(data) < 4:
            return None, 0
        
        for sid, s in enumerate(data):
            if len(s) < 20 or len(s) > 300:
                continue
            
            if "?" not in s:
                continue

            s = s.strip(string.punctuation)
            s = s + "?"

            d = word_tokenize(s)
            
            # noun
            tag = nltk.pos_tag(d)
            noun_pos = [i for i, t in enumerate(tag) if t[1] in ["NNP", "NNPS", "NN", "NNS"]]
            if len(noun_pos) < 3:
                continue
            
            origin_noun_pos = copy.deepcopy(noun_pos)
            random.shuffle(noun_pos)
            origin_d = copy.deepcopy(d)
            for origin_p, p in zip(origin_noun_pos, noun_pos):
                d[p] = origin_d[origin_p]
            
            # antonym
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
            
            if len(cands) == 0:
                continue        
             
            mask_p, mask_w, mask_ant = random.choice(cands)
            
            d[mask_p] = mask_ant
            
            new_d = self.tokenizer.detokenize(d)
            origin_d = self.tokenizer.detokenize(origin_d)

            all_data.append({
                "pos": origin_d,
                "neg": new_d
            })
            
        return all_data, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="merge.txt", help='Path to input TXT')
    parser.add_argument("--output_dir", default="pseudo_data_tmp/para", type=str)
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
    with open(os.path.join(args.output_dir, "question.jsonl"), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    pool.close()

if __name__ == '__main__':
    main()