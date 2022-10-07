import argparse
import multiprocessing
import os
import string
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


class MWPEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        MWPEncoder.tokenizer = EncDecTokenizer(os.path.join(self.args.tokenizer_path, 'spiece.model'))
        all_prompts = [
            ("a-what-fill", ["Read the document:", "What should you fill in the", "?"], ["", ""]),
            ("a-replace", ["", "Replace the", "in the above passage with the correct phrase"], ["", ""]),
            ("a-refer-to", ["", "What does the", "in the above passage refer to?"], ["", ""]),
            ("a-fill-in", ["", "Read the above document and fill in", ":"], ["", ""]),
            ("a-input-output", ["Input:", "Guess what is in", ":"], ["", ""]),
            ("b-refer-to", ["What does the", "in the next passage refer to?", ""], ["", ""]),
            ("b-fill-in", ["Fill in", "in the next passage.", ""], ["", ""]),
            ("b-generate", ["Generate a possible answer for the blank", "in the next document.", ""], ["", ""]),
            ("b-stand-for", ["What does", "stands for in the next document?", ""], ["", ""]),
            ("b-homework", ["Homework: find the best phrase for", "in the passage:", ""], ["", ""]),
        ]
        MWPEncoder.all_prompts = [(x[0], [self.tokenizer.encode(xx) for xx in x[1]], [self.tokenizer.encode(xx) for xx in x[2]]) for x in all_prompts]


    def encode_one_prompt(self, doc, prompt, mask_tokens):
        _, context_prompt, target_prompt = prompt
        ctx_prompt_token_num = len(context_prompt[0]) + len(context_prompt[1])
        
        paras = doc.split("<@x(x!>")
        doc = " ".join(paras)
        sents = sent_tokenize(doc)
        context_tokens = []
        target_tokens = []
        for i in range(len(sents)):
            tokens = self.tokenizer.encode(sents[i])
            if len(context_tokens) + len(tokens) + ctx_prompt_token_num < 505:
                context_tokens.extend(tokens)

        if len(context_tokens) < 50:
            return None
        
        if self.tokenizer.decode(context_tokens[-1:])[-1] not in string.punctuation:
            context_tokens.append(5)

        length = random.randint(1, 20)
        begin = random.randint(0, len(context_tokens) - 20 - 1)
        target_tokens = context_tokens[begin:begin+length]

        surr_begin = max(0, begin - 20)
        surr_end = min(begin+length, len(context_tokens))
        surrounding = self.tokenizer.decode(context_tokens[surr_begin:surr_end])
        if "https" in surrounding or "http" in surrounding or ".com" in surrounding:
            return None

        context_tokens = context_tokens[:begin] + mask_tokens + context_tokens[begin+length:]

        target_str = self.tokenizer.decode(target_tokens)
        context_tokens = context_prompt[0] + context_tokens + context_prompt[1]
        target_tokens = [0] + target_prompt[0] + target_tokens + target_prompt[1]

        assert len(context_tokens) < 512
        assert len(target_tokens) < 127

        return (context_tokens, target_tokens, target_str)

    def encode(self, doc):
        # print(doc[:20])
        # end with <eod>
        mask_list = ["_", "__", "___", "@@@", "(())", "$$$", "%%%", "###", "***", "+++"]
        pid = random.randint(0, len(self.all_prompts)-1)
        mask_tokens = self.tokenizer.encode(random.choice(mask_list))
        prompt_name, context_prompt, target_prompt = self.all_prompts[pid]
        if prompt_name[0] == "a":
            prompt = (prompt_name, [context_prompt[0], context_prompt[1] + mask_tokens + context_prompt[2]], target_prompt)
        else:
            prompt = (prompt_name, [context_prompt[0] + mask_tokens + context_prompt[1], context_prompt[2]], target_prompt)

        res = self.encode_one_prompt(doc, prompt, mask_tokens)
                
        return (res, pid), len(doc)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default="/home/yourname/data_en/pretrain_merge/split/merge_shuf_1.txt", help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/home/yourname/UDIT/vocab_en", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/home/yourname/UDIT/self_sup_data/selfsup/merge/mwp", type=str)

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


def mwp_process(args, encoded_docs, tokenizer, split, sample_num):
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


def main():
    args = get_args()
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = MWPEncoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
    all_samples = mwp_process(args, encoded_docs, tokenizer, args.split, args.sample_num)

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "cache_{}_{}.pkl".format(args.split, args.sample_num)), "wb") as f:
        pickle.dump(all_samples, f)

    pool.close()


if __name__ == '__main__':
    main()