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

import json


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        pass

    def encode(self, all_lines):
        # end with <eod>
        all_data = []

        pos_neg_data = []

        for i, line in enumerate(all_lines):
            if len(line) > 5000000:
                return None, 0

            # data
            data = line.strip().replace("\u201d", "\"").replace("\u201c", "\"").replace(
                    "\u2019", "\'").replace("\u2018", "\'").replace("\u2014", "-").replace("\u2013", "-").replace("``", "\"").replace("''", "\"")

            if "https" in data or "http" in data or ".com" in data:
                return None, 0

            if re.search(r"\d{6}", data):
                return None, 0

            data = data.split("<@x(x!>")
            data = [re.sub(r"\(.*\)", "", d) for d in data]
            pos_neg_data.append(data)
        
        pos_data = pos_neg_data[0]
        neg_data = pos_neg_data[1:]

        for did, d in enumerate(pos_data):
            e = len(d) - 1
            while e >= 0:
                if d[e] == "?":
                    break
                e -= 1
            if e <= 0:
                continue
            s = e - 1
            while s >= 0:
                if d[s] in ".\"?!":
                    break
                s -= 1
            s += 1
            question = d[s:e+1]
            if did == 0 and s == 0:
                continue
            elif s == 0:
                context = pos_data[did-1]
            elif did == 0:
                context = d[0:s]
            else:
                context = pos_data[did-1] + d[0:s]

            ee = len(context) - 1
            while ee >= 0:
                if context[ee] in ".!":
                    break                    
                ee -= 1

            if ee <= 0:
                continue
            context = context[:ee+1]

            if did == len(pos_data) - 1 and e == len(d) - 1:
                continue
            elif e == len(d) - 1:
                answer = pos_data[did + 1]
            elif did == len(pos_data) - 1:
                answer = d[e+1:]
            else:
                answer = d[e+1:] + pos_data[did + 1]
            
            answer.replace("READ MORE", "")

            if did + 2 >= len(pos_data):
                continue

            hard_cand = pos_data[did+2]
            
            easy_cand = []
            for neg in neg_data:
                r = random.randint(0, len(neg) - 1)
                easy_cand.append(neg[r])

            all_cands_tmp_tmp = [answer] + [hard_cand] + easy_cand
            all_cands_tmp = []
            for cand in all_cands_tmp_tmp:
                s = 0
                while s < len(cand):
                    if cand[s] not in "\".,!@#$%^&*()[]:;?/|\\-_=+`~ ":
                        break
                    s += 1
                all_cands_tmp.append(cand[s:])

            if min([len(x) for x in all_cands_tmp]) < 10 or len(context) < 30 or len(question) < 10 or len(context) > 1000:
                continue

            all_cands = []
            for op in all_cands_tmp:
                op = op.split(".")
                op_tmp = op[0]
                j = 1
                while j < len(op) and len(op_tmp) < 150:
                    op_tmp += "." + op[j]
                    j += 1
                if re.match(r"[a-zA-Z0-9]", op_tmp[-1]) is not None:
                    op_tmp = op_tmp + "."
                else:
                    op_tmp = op_tmp[:-1] + "."
                all_cands.append(op_tmp) 

            if any(["?" in x for x in all_cands]):
                continue

            new_answer = all_cands[0]
            random.shuffle(all_cands)
            label = all_cands.index(new_answer)   

            assert len(all_cands) == self.args.option_num

            all_data.append({
                "context": context,
                "question": question,
                "options": all_cands,
                "label": label,
                "answer": new_answer
            })
        
        return all_data, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="merge.txt", help="Path to input TXT")
    parser.add_argument("--input_neg1", type=str, default="merge_shuf_1.txt", help="Path to input TXT")
    parser.add_argument("--input_neg2", type=str, default="merge_shuf_2.txt", help="Path to input TXT")
    parser.add_argument("--option_num", type=int, default=4)
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
    fin_neg1 = open(args.input_neg1, "r", encoding='utf-8')
    fin_neg2 = open(args.input_neg2, "r", encoding='utf-8')

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    if args.option_num == 4:
        encoded_docs = pool.imap_unordered(encoder.encode, zip(fin, fin_neg1, fin_neg2), 10)
    elif args.option_num == 3:
        encoded_docs = pool.imap_unordered(encoder.encode, zip(fin, fin_neg1), 10)
    else:
        raise ValueError("Invalid option number. Choose between 3 and 4.")

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
    with open(os.path.join(args.output_dir, "natural_question_option{}.jsonl".format(args.option_num)), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    pool.close()

if __name__ == '__main__':
    main()