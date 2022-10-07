import os
import hashlib
import json
import random
from tqdm import tqdm

random.seed(981217)

DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
# acceptable ways to end a sentence
END_TOKENS = [".", "!", "?", "...", "'", "`", '"', DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ")"]


def _read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def _get_art_abs(story_file):
    """Get abstract (highlights) and article from a story file path."""
    # Based on https://github.com/abisee/cnn-dailymail/blob/master/
    #     make_datafiles.py

    lines = _read_text_file(story_file)

    # The github code lowercase the text and we removed it in 3.0.0.

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't end in
    # periods; consequently they end up in the body of the article as run-on
    # sentences)
    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)
    abstract = " ".join(highlights)

    return article, abstract


def _get_hash_from_path(p):
    """Extract hash from path."""
    basename = os.path.basename(p)
    return basename[0: basename.find(".story")]


def _find_files(data_dir, publisher, url_dict):
    """Find files corresponding to urls."""
    if publisher == "cnn":
        top_dir = os.path.join(data_dir, "cnn", "stories")
    elif publisher == "dm":
        top_dir = os.path.join(data_dir, "dailymail", "stories")
    else:
        raise ValueError("Unsupported publisher")
    files = sorted(os.listdir(top_dir))

    ret_files = []
    for p in files:
        if _get_hash_from_path(p) in url_dict:
            ret_files.append(os.path.join(top_dir, p))
    return ret_files


def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        u = u.encode("utf-8")
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}


data_dir = "/home/yourname/data_en/cnn_dm"

for split in ["train", "val"]:
    urls = _get_url_hashes(os.path.join(data_dir, "all_{}.txt".format(split)))
    cnn = _find_files(data_dir, "cnn", urls)
    dm = _find_files(data_dir, "dm", urls)
    all_files = cnn + dm
    
    data = []
    for file_path in tqdm(all_files):
        article, summary = _get_art_abs(file_path)
        data.append({
            "text": article,
            "summary": summary,
            "answer": summary,
        })
    
    random.shuffle(data)
    with open(os.path.join(data_dir, "{}.jsonl".format(split)), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

os.system("mv {} {}".format(os.path.join(data_dir, "val.jsonl"), os.path.join(data_dir, "valid.jsonl")))
os.system("cp {} {}".format(os.path.join(data_dir, "valid.jsonl"), os.path.join(data_dir, "test.jsonl")))
