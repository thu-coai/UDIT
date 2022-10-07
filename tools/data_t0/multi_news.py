# Lint as: python3
"""Multi-News dataset."""


import os

import datasets


_CITATION = """
@misc{alex2019multinews,
    title={Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model},
    author={Alexander R. Fabbri and Irene Li and Tianwei She and Suyi Li and Dragomir R. Radev},
    year={2019},
    eprint={1906.01749},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
Multi-News, consists of news articles and human-written summaries
of these articles from the site newser.com.
Each summary is professionally written by editors and
includes links to the original articles cited.

There are two features:
  - document: text of news articles seperated by special token "|||||".
  - summary: news summary.
"""

_URL = "https://drive.google.com/uc?export=download&id=1vRY2wM6rlOZrf9exGTm5pXj5ExlVwJ0C"

_DOCUMENT = "document"
_SUMMARY = "summary"


class MultiNews(datasets.GeneratorBasedBuilder):
    """Multi-News dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({_DOCUMENT: datasets.Value("string"), _SUMMARY: datasets.Value("string")}),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage="https://github.com/Alex-Fabbri/Multi-News",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extract_path = os.path.join("/home/yourname/data_hf/multi_news", "multi-news-original")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(extract_path, "train")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(extract_path, "val")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(extract_path, "test")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(os.path.join(path + ".src"), encoding="utf-8") as src_f, open(
            os.path.join(path + ".tgt"), encoding="utf-8"
        ) as tgt_f:
            for i, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                yield i, {
                    # In original file, each line has one example and natural newline
                    # tokens "\n" are being replaced with "NEWLINE_CHAR". Here restore
                    # the natural newline token to avoid special vocab "NEWLINE_CHAR".
                    _DOCUMENT: src_line.strip().replace("NEWLINE_CHAR", "\n"),
                    # Remove the starting token "- " for every target sequence.
                    _SUMMARY: tgt_line.strip().lstrip("- "),
                }
