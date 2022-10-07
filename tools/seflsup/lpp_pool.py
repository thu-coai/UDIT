import argparse
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
import time
import torch
import random
import numpy as np
import pickle
import random
from nltk.tokenize import sent_tokenize

from tokenization_t5 import EncDecTokenizer

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(object):
    def __init__(self, args, function_words):
        self.args = args
        self.function_words = function_words

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = EncDecTokenizer(os.path.join(self.args.tokenizer_path, 'spiece.model'))

    def encode(self, doc):
        phrase_pool = {k:set() for k in self.function_words}
        
        paras = doc.split("<@x(x!>")
        doc = " ".join(paras)
        sents = sent_tokenize(doc)
        for sent in sents:
            if len(sent) == 0 or sent[-1] != ".":
                continue
            words = sent.split(" ")
            func_words_poses = []
            for wid in range(len(words) // 2, len(words)):
                w = words[wid]
                if w in self.function_words:
                    func_words_poses.append(wid)
            
            if len(func_words_poses) > 0:
                pos = func_words_poses[-1]
                phrase_pool[words[pos]].add(" ".join(words[pos:]))

        return phrase_pool, len(doc)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default="/home/yourname/data_en/pretrain_merge/merge.txt", help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/home/yourname/UDIT/vocab_en", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/home/yourname/UDIT/self_sup_data/selfsup/merge/", type=str)

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--split', default="validation")
    group.add_argument('--min_sample_num', default=5000)
    group.add_argument('--max_sample_num', default=5000000)
    

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    function_words = {
        "about", "above", "as", "after", "a", "an", "at",
        "and", "are", "be", "been", "before",
        "by", "can", "could", "during",
        "except", "from", "for", "have", "had", "including", "in", "into",
        "is", "must", "may", "might", "nor", "on", "or", "onto", "of",
        "to", "then", "the", "than", "should", "under", "using",
        "were", "was", "would", "will", "with", "within", "without",
    }

    phrase_pool = {k: set() for k in function_words}

    encoder = Encoder(args, function_words)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    print(f"Vocab size: {tokenizer.vocab_size}")
    
    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    print("tokenizer vocab size:", tokenizer.vocab_size)
    sid = 0
    for lid, (phrases, bytes_processed) in enumerate(encoded_docs, start=1):
        
        total_bytes_processed += bytes_processed
        if phrases is None:
            continue

        # print(min([len(x) for x in phrase_pool.values()]))
        # print(min(phrase_pool, key=lambda k: len(phrase_pool[k])))

        if min([len(x) for x in phrase_pool.values()]) >= args.min_sample_num:
            break

        for k in phrases:
            if len(phrase_pool[k]) < args.max_sample_num:
                phrase_pool[k].update(phrases[k])

        if lid % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents",
                  f"({lid/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
            print(min([len(x) for x in phrase_pool.values()]), min(
                phrase_pool, key=lambda k: len(phrase_pool[k])))
            print(max([len(x) for x in phrase_pool.values()]), max(
                phrase_pool, key=lambda k: len(phrase_pool[k])))

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "lpp_pool_{}_{}.pkl".format(args.min_sample_num, args.max_sample_num)), "wb") as f:
        pickle.dump(phrase_pool, f)

    pool.close()

if __name__ == '__main__':
    main()