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

from tokenization_t5 import EncDecTokenizer
from nsg import NSGEncoder, nsg_process
from mwp import MWPEncoder, mwp_process
from lpp_gen import LPPGenEncoder, lpp_gen_process
from lpp_cls import LPPCLSEncoder, lpp_cls_process

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default="/home/yourname/data_en/pretrain_merge/merge.txt", help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/home/yourname/UDIT/vocab_en", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/home/yourname/UDIT/self_sup_data/selfsup/merge_total/", type=str)

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0

    return args

def main():
    args = get_args()
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))

    all_sample_num = {
        "train": 10,
        "validation": 10
    }

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    for split in ["train", "validation"]:
        for name, encoder_cls, process_fn in [
            ("nsg", NSGEncoder, nsg_process),
            ("mwp", MWPEncoder, mwp_process),
            # ("lpp_gen", LPPGenEncoder, lpp_gen_process),
            # ("lpp_cls", LPPCLSEncoder, lpp_cls_process)
        ]:
            print(name)
            sample_num = all_sample_num[split]
            encoder = encoder_cls(args)
            pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
            encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
            all_samples, enc_sizes, dec_sizes, cand_sizes = process_fn(args, encoded_docs, tokenizer, split, sample_num)
            
            output_path = os.path.join(args.output_path, name)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "cache_{}_{}.pkl".format(split, sample_num)), "wb") as f:
                pickle.dump((all_samples, enc_sizes, dec_sizes, cand_sizes), f)

            pool.close()

if __name__ == '__main__':
    main()