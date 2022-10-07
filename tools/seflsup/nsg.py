import argparse
import multiprocessing
import os
import sys
import string
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


class NSGEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        NSGEncoder.tokenizer = EncDecTokenizer(os.path.join(self.args.tokenizer_path, 'spiece.model'))
        all_prompts = [
            ("input-output", ["Input:", "Write the next sentence:"], ["", ""]),
            ("continuation-document", ["", "What is a possible continuation for the document?"], ["", ""]),
            ("gen-ending", ["Generate a possible next sentence for the following story:", ""], ["", ""]),
            ("read-gen", ["Read the following passage:", "Generate a possible ending for this passage:"], ["", ""]),
            ("possible-continuation", ["Read the document:", "What do you think is the most probable continuation?"], ["", ""]),
            ("how-next", ["", "How does this story go next?"], ["", ""]),
            ("task-is", ["The task is to generate a following sentence for the document:", ""], ["", ""]),
            ("begin-like", ["If an article begins like this:", "How does it continue?"], ["", ""]),
            ("appropriate", ["What is an appropriate continuation of the following text:", ""], ["", ""]),
            ("write", ["Write the more likely continuation for the following passage:", ""], ["", ""])
        ]
        NSGEncoder.all_prompts = [(x[0], [self.tokenizer.encode(xx) for xx in x[1]], [self.tokenizer.encode(xx) for xx in x[2]]) for x in all_prompts]

    def encode_one_prompt(self, doc, prompt):
        if len(doc) < 30:
            return None
        prompt_name, context_prompt, target_prompt = prompt
        ctx_prompt_token_num = len(context_prompt[0]) + len(context_prompt[1])
        tgt_prompt_token_num = len(target_prompt[0]) + len(target_prompt[1])
        
        paras = doc.split("<@x(x!>")
        doc = " ".join(paras)
        sents = sent_tokenize(doc)
        context_tokens = []
        target_tokens = []
        for i in range(len(sents)):
            tokens = self.tokenizer.encode(sents[i])
            if i == len(sents) - 1:
                lim = 126 - tgt_prompt_token_num
                if len(tokens) > lim:
                    target_tokens = tokens[:lim]
                else:
                    target_tokens = tokens + [1]
            else:
                if len(context_tokens) + len(tokens) + ctx_prompt_token_num < 510:
                    context_tokens.extend(tokens)
                else:
                    lim = 127 - tgt_prompt_token_num
                    if len(tokens) >= lim:
                        target_tokens = tokens[:lim]
                    else:
                        target_tokens = tokens + [1]
                    break

        if len(context_tokens) == 0:
            return None

        if self.tokenizer.decode(context_tokens[-1:])[-1] not in string.punctuation:
            context_tokens.append(5)

        target_str = self.tokenizer.decode(target_tokens)
        if "https" in target_str or "http" in target_str or ".com" in target_str:
            return None

        context_tokens = context_prompt[0] + context_tokens + context_prompt[1]
        target_tokens = [0] + target_prompt[0] + target_tokens + target_prompt[1]
        
        assert len(context_tokens) < 512
        assert len(target_tokens) < 129, len(target_tokens)
        
        return (context_tokens, target_tokens, target_str)

    def encode(self, doc):
        # end with <eod>
        pid = random.randint(0, len(self.all_prompts)-1)
        prompt = self.all_prompts[pid]
        res = self.encode_one_prompt(doc, prompt)
                
        return (res, pid), len(doc)


def nsg_process(args, encoded_docs, tokenizer, split, sample_num):
    all_samples = []
    
    proc_start = time.time()
    total_bytes_processed = 0
    sid = 0
    for lid, (s, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        data, pid = s
        if data is None:
            continue
        if sample_num != -1 and sid >= sample_num:
            break

        context, target, target_str = data

        all_samples.append({
            "idxs": [pid, lid, sid],
            "enc_input_ids": context,
            "dec_input_ids": target[:-1],
            "label_ids": target[1:],
            "answer": target_str,
            "options": None,
            "cands": None
        })

        sid += 1

        if lid % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents",
                  f"({lid/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    random.shuffle(all_samples)

    return all_samples


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default="/home/yourname/data_en/pretrain_merge/split/merge_shuf_0.txt", help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/home/yourname/UDIT/vocab_en", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/home/yourname/UDIT/self_sup_data/selfsup/merge/nsg", type=str)

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--split', default="all")
    group.add_argument('--sample_num', default=-1)

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0

    return args


def main():
    args = get_args()
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = NSGEncoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
    all_samples = nsg_process(args, encoded_docs, tokenizer, args.split, args.sample_num)

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "cache_{}_{}.pkl".format(args.split, args.sample_num)), "wb") as f:
        pickle.dump(all_samples, f)

    pool.close()


if __name__ == '__main__':
    main()