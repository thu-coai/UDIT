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


class LPPGenEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        LPPGenEncoder.tokenizer = EncDecTokenizer(os.path.join(self.args.tokenizer_path, 'spiece.model'))
        all_prompts = [
            ("gen-next-phrase", ["Generate the next phrase of the following passage:", "", ""], ["", ""]),
            ("?-answer-question", ["Input:", "Question:", "Anaswer:"], ["", ""]),
            ("?-read-answer", ["Read the following context and answer the question.", "Question:", ""], ["", ""]),
            ("?-according-answer", ["", "According to the above context, answer the following question:", ""], ["", ""]),
            ("?-gen-best", ["", "If I ask you", "Give me the best answer:"], ["", ""]),
            ("gen-continuation", ["Write the likely continuation to the following passage:", "", ""], ["", ""]),
            ("?-gen-plausible", ["", "Generate the most plausible answer for the following question:", ""], ["", ""]),
            ("?-best-answer", ["", "If we ask:", "What's the best answer?"], ["", ""]),
            ("end", ["How does this document end?", "", ""], ["", ""]),
            ("?-exercise", ["", "Exercise: what is the answer to the following question?", ""], ["", ""])
        ]
        LPPGenEncoder.all_prompts = [(x[0], [self.tokenizer.encode(xx) for xx in x[1]], [self.tokenizer.encode(xx) for xx in x[2]]) for x in all_prompts]
        LPPGenEncoder.function_words = {
            "about", "above", "as", "after", "a", "an", "at",
            "and", "are", "be", "been", "before", 
            "by", "can", "could", "during", "excluding",
            "except", "from", "for", "have", "had", "including", "in", "into", 
            "is", "must", "may", "might", "neither", "nor", "on", "or", "onto", "of",
            "shall", "to", "then", "the", "than", "should", "under", "using",
            "were", "was", "would", "will", "with", "within", "without", 
        }

    def encode_one_prompt(self, doc, prompt):
        prompt_name, context_prompt, target_prompt = prompt
        ctx_prompt_token_num = sum([len(x) for x in context_prompt])
        tgt_prompt_token_num = sum([len(x) for x in target_prompt])
        
        paras = doc.split("<@x(x!>")
        doc = " ".join(paras)
        sents = sent_tokenize(doc)
        context_tokens_origin = []
        context_tokens = []
        question_tokens = []
        target_tokens = []
        for i in range(len(sents)):
            tokens = self.tokenizer.encode(sents[i])
            if sum([len(x) for x in context_tokens_origin]) + len(tokens) + ctx_prompt_token_num < 506:
                context_tokens_origin.append(tokens)

        for sid in range(len(context_tokens_origin)-1, -1, -1):
            tokens = context_tokens_origin[sid]
            sent = self.tokenizer.decode(tokens)
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
                context_tokens = [x for y in context_tokens_origin[:sid] for x in y]
                question_tokens = self.tokenizer.encode(" ".join(words[:pos]))
                target_tokens = self.tokenizer.encode(" ".join(words[pos:]))
                if len(target_tokens) + tgt_prompt_token_num < 127:
                    break
                else:
                    context_tokens, question_tokens, target_tokens = [], [], []
         
        if len(context_tokens) == 0 or len(question_tokens) == 0 or len(target_tokens) == 0:
            return None
                
        target_str = self.tokenizer.decode(target_tokens)
        
        if "https" in target_str or "http" in target_str or ".com" in target_str:
            return None
        
        if "\"" in target_str or "\u201d" in target_str or "\u201c" in target_str:
            return None
        
        if prompt_name[0] == "?":
            question_tokens.append(58)
        context_tokens_2 = context_prompt[0] + context_tokens + context_prompt[1] + question_tokens + context_prompt[2]
        target_tokens = [0] + target_prompt[0] + target_tokens + target_prompt[1] + [1]

        if len(context_tokens_2) >= 512 or len(target_tokens) >= 127:
            # too much unk
            return None

        assert len(context_tokens_2) < 512
        assert len(target_tokens) < 127

        return (context_tokens_2, target_tokens, target_str)

    def encode(self, doc):
        # end with <eod>
        pid = random.randint(0, len(self.all_prompts)-1)
        prompt = self.all_prompts[pid]
        res = self.encode_one_prompt(doc, prompt)
                
        return (res, pid), len(doc)


def lpp_gen_process(args, encoded_docs, tokenizer, split, sample_num):
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
    group.add_argument('--input', type=str, default="/home/yourname/data_en/pretrain_merge/split/merge_shuf_2.txt", help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/home/yourname/UDIT/vocab_en", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/home/yourname/UDIT/self_sup_data/selfsup/merge/lpp_gen", type=str)

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

    encoder = LPPGenEncoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
    all_samples = lpp_gen_process(args, encoded_docs, tokenizer, args.split, args.sample_num)

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "cache_{}_{}.pkl".format(args.split, args.sample_num)), "wb") as f:
        pickle.dump(all_samples, f)

    pool.close()


if __name__ == '__main__':
    main()