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


class LPPCLSEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        LPPCLSEncoder.tokenizer = EncDecTokenizer(os.path.join(self.args.tokenizer_path, 'spiece.model'))
        all_prompts = [
            ("?-cqa-Input-Output", ["Input:", "Question:", "Answer:", ""], ["Yes", "No"]),
            ("?-caq-can-answer", ["", "Can the phrase", "answer the following question:", ""], ["Yes", "No"]),
            ("?-caq-is-answer", ["Given the passage:", "The phrase", "is the answer to the question:", "True or False?"], ["True", "False"]),
            ("*-cqa-does-follow", ["Given that:", "Does it follow that", "", "Yes or no?"], ["Yes", "No"]),
            ("*-cqa-base-know", ["", "After reading the previous passage, will you continue to write", "", "True or false?"], ["True", "False"]),
            ("?-cqa-always-never", ["", "Based on the previous document, we ask", "Is the answer", "always or never true?"], ["Always", "Never"]),
            ("_-cqa-base-know", ["", "Given the previous passage, we can say that", "", "Correct or incorrect?"], ["Correct", "Incorrect"]),
            ("?-qac-judge", ["Question:", "We have an answer", "Judge the correctness of the answer according to the following passage:", ""], ["Correct", "Incorrect"]),
            ("_-cqa-guaranteed", ["", "Is the statement", "", "a possible following?"], ["Yes", "No"]),
            ("_-qac-belong", ["Does the sentence", "", "belong to the same document of the following passage?", ""], ["Yes", "No"])
        ]
        LPPCLSEncoder.all_prompts = [(x[0], [self.tokenizer.encode(xx) for xx in x[1]], [self.tokenizer.encode(xx) for xx in x[2]]) for x in all_prompts]
        LPPCLSEncoder.function_words = {
            "about", "above", "as", "after", "a", "an", "at",
            "and", "are", "be", "been", "before",
            "by", "can", "could", "during",
            "except", "from", "for", "have", "had", "including", "in", "into",
            "is", "must", "may", "might", "nor", "on", "or", "onto", "of",
            "to", "then", "the", "than", "should", "under", "using",
            "were", "was", "would", "will", "with", "within", "without",
        }
        with open("/home/yourname/UDIT/self_sup_data/selfsup/merge/lpp_pool_5000_5000000.pkl", "rb")as f:
            LPPCLSEncoder.phrase_pool = pickle.load(f)
            for k in LPPCLSEncoder.phrase_pool:
                LPPCLSEncoder.phrase_pool[k] = list(LPPCLSEncoder.phrase_pool[k])
            print("Load pool end.")

    def encode_one_prompt(self, doc, prompt):
        prompt_name, context_prompt, label_prompt = prompt
        ctx_prompt_token_num = sum([len(x) for x in context_prompt])
        
        paras = doc.split("<@x(x!>")
        doc = " ".join(paras)
        sents = sent_tokenize(doc)
        context_tokens_origin = []
        context_tokens = []
        question_tokens = []
        answer_tokens = []
        target_tokens = []
        for i in range(len(sents)):
            tokens = self.tokenizer.encode(sents[i])
            if len(context_tokens_origin) + len(tokens) + ctx_prompt_token_num < 511:
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
                question_str = " ".join(words[:pos])
                if random.random() < 0.5:
                    answer_str = " ".join(words[pos:])
                    target_tokens = label_prompt[0]
                else:
                    truth = " ".join(words[pos:])
                    answer_str = truth
                    while answer_str == truth:
                        answer_str = random.choice(self.phrase_pool[words[pos]])
                    target_tokens = label_prompt[1]
                if prompt_name[0] == "?":
                    question_str += "?"
                    answer_str = "\"" + answer_str + "\""
                elif prompt_name[0] == "*":
                    question_str = "\"" + question_str
                    answer_str = answer_str + "\"?"
                else: # "_"
                    question_str = "\"" + question_str
                    answer_str = answer_str + "\""
                    
                question_tokens = self.tokenizer.encode(question_str)
                answer_tokens = self.tokenizer.encode(answer_str)
                break
        
        if len(context_tokens) == 0 or len(question_tokens) == 0 or len(target_tokens) == 0:
            return None
                
        target_str = self.tokenizer.decode(target_tokens)
        if prompt_name[2:5] == "cqa":
            context_tokens = context_prompt[0] + context_tokens + context_prompt[1] + question_tokens + context_prompt[2] + answer_tokens + context_prompt[3]
        elif prompt_name[2:5] == "caq":
            context_tokens = context_prompt[0] + context_tokens + context_prompt[1] + answer_tokens + context_prompt[2] + question_tokens + context_prompt[3]
        else: # qac
            context_tokens = context_prompt[0] + question_tokens + context_prompt[1] + answer_tokens + context_prompt[2] + context_tokens + context_prompt[3]
            
        target_tokens = [0] + target_tokens + [1]
        
        if len(context_tokens) > 511 or len(target_tokens) > 127:
            return None
        
        assert len(context_tokens) < 512
        assert len(target_tokens) < 127
        
        label_prompt_str = [self.tokenizer.decode(x) for x in label_prompt]
        label_prompt_tokens = [[0] + x + [1] for x in label_prompt]

        return (context_tokens, target_tokens, target_str, label_prompt_tokens, label_prompt_str)

    def encode(self, doc):
        # end with <eod>
        pid = random.randint(0, len(self.all_prompts)-1)
        prompt = self.all_prompts[pid]
        res = self.encode_one_prompt(doc, prompt)

        return (res, pid), len(doc)


def lpp_cls_process(args, encoded_docs, tokenizer, split, sample_num):
    all_samples_label = [[], []]

    if sample_num != -1:
        sample_num = sample_num // len(all_samples_label)

    proc_start = time.time()
    total_bytes_processed = 0

    sid = 0
    for lid, (s, bytes_processed) in enumerate(encoded_docs, start=1):
        
        total_bytes_processed += bytes_processed
        data, pid = s
        if data is None:
            continue
    
        if sample_num != -1 and min([len(x) for x in all_samples_label]) >= sample_num:
            break

        context, target, target_str, cands, options = data

        label = options.index(target_str)
        if sample_num != -1 and len(all_samples_label[label]) >= sample_num:
            continue

        all_samples_label[label].append({
            "idxs": [pid, lid, sid],
            "enc_input_ids": context,
            "dec_input_ids": target[:-1],
            "label_ids": target[1:],
            "answer": target_str,
            "options": options if split != "train" else None,
            "cands": {
                "input_ids": [c[:-1] for c in cands],
                "target_ids": [c[1:] for c in cands],
                "label": options.index(target_str),
            } if split != "train" else None
        })

        sid += 1

        if lid % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents",
                  f"({lid/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    min_len = min([len(x) for x in all_samples_label])
    all_samples = all_samples_label[0][:min_len] + all_samples_label[1][:min_len]
    
    random.shuffle(all_samples)

    return all_samples


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default="/home/yourname/data_en/pretrain_merge/split/merge_shuf_3.txt", help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/home/yourname/UDIT/vocab_en", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/home/yourname/UDIT/self_sup_data/selfsup/merge/lpp_cls", type=str)

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

    encoder = LPPCLSEncoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
    all_samples = lpp_cls_process(args, encoded_docs, tokenizer, args.split, args.sample_num)

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "cache_{}_{}.pkl".format(args.split, args.sample_num)), "wb") as f:
        pickle.dump(all_samples, f)

    pool.close()


if __name__ == '__main__':
    main()