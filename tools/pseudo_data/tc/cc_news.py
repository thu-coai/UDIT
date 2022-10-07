import re
import os
import sys
import json
import time
import random
import argparse
import multiprocessing


from nltk import sent_tokenize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
print(sys.path)

from tokenization_t5 import EncDecTokenizer

class Encoder(object):
    def __init__(self, args, topics):
        self.args = args
        self.topics = topics

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = EncDecTokenizer(self.args.tokenizer_path)
        Encoder.banned_list = ["news", "article", "story", "stories",
                               "articles", "wires", "en", "post", "content", "local-news", "reuters", "english",
                               "eng", "News", "_news", "uk", "canada", "us", "news-releases", "au", "india", "national",
                               "local", "tag", "a", "whats-on", "india-news", "fantasy"]

        Encoder.name_topic_map = {name: top for top in self.topics for name in self.topics[top]}

    def clean_url(self, url):
        if url is None:
            return None
        url = url.strip()
        m = re.match(r"http.+?//.+?/(.+?)/.*", url)
        if m is not None:
            if m.group(1) in self.banned_list:
                mm = re.match(r"http.+?//.+?/(.+?)/(.+?)/.*", url)
                if mm is not None:
                    if mm.group(2) in self.banned_list:
                        mmm = re.match(r"http.+?//.+?/(.+?)/(.+?)/(.+?)/.*", url)
                        if mmm is not None:
                            name = mmm.group(3)
                        else:
                            return None
                    else:
                        name = mm.group(2)
                else:
                    return None
            else:
                name = m.group(1)

            return name

        else:
            return None

    def encode(self, line):
        data = json.loads(line)
        url = data["url"]
        name = self.clean_url(url)
        sample = None
        if name is not None and name in self.name_topic_map:
            title = data["title"]
            text = data["text"]
            if title is not None:
                text = title + "\n" + text
            sents = sent_tokenize(text)
            tokens_num = 0
            new_sents = []
            for sent in sents:
                tokens = self.tokenizer.encode(sent)
                if tokens_num + len(tokens) < 450:
                    new_sents.append(sent)
                else:
                    break
            text = " ".join(new_sents)
            topic = self.name_topic_map[name]
            sample = (text, topic)

        return sample, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="cc_news/cc_news_url.jsonl", help="Path to input TXT")
    parser.add_argument("--tokenizer_path", type=str, default="vocab_en/spiece.model", help="Vocab path of the T5 tokenizer")
    parser.add_argument("--output_dir", default="pseudo_data_tmp/tc", type=str)
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


def main():
    args = get_args()
    startup_start = time.time()

    set_random_seed(20)

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    topics = {
        "sport": ["sports", "sport", "sports-story", "nba", "football", "soccer", "basketball"],
        "world": ["world"],
        "business": ["business", "stocks", "markets", "money"],
        "entertainment": ["entertainment", "arts-and-entertainment"],
        "politics": ["politics", "politics-government"],
        "health": ["health"],
        "city": ["city"],
        "life": ["lifestyle", "life"],
        "technology": ["technology", "tech", "science"],
        "crime": ["crime"],
        "music": ["music"],
        "arts": ["arts"],
        "movies": ["movies", "bollywood"],
        "education": ["education"]
    }

    encoder = Encoder(args, topics)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)
    all_samples = {k:[] for k in topics}

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    sample_num = 0
    for i, (sample, bytes_processed) in enumerate(encoded_docs, start=1):
        if sample is None:
            continue
        total_bytes_processed += bytes_processed
        
        text, topic = sample
        all_samples[topic].append({
            "text": text,
            "topic": topic
        })
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print("Processed {} documents, ({} docs/s, {} MB/s). {} Samples".format(
                i, i/elapsed, mbs, sample_num), file=sys.stderr)
        
        sample_num += 1

    print({k:len(all_samples[k]) for k in all_samples})

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "cc_news.json"), "w") as f:
        f.write(json.dumps(all_samples) + "\n")



if __name__ == "__main__":
    main()