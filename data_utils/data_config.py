import os

BASE_DATA_DIR = ""

def string_to_float(preds, labels):
    return [float(p) for p in preds], [float(l) for l in labels]


DATA_GROUP_CONFIG = {
    "MCQA": ["dream", "quail", "quartz", "social_i_qa", "wiqa", "cosmos_qa", "qasc", "quarel", "sciq", "wiki_hop"],
    "MCQA_PSEUDO": ["quail_pseudo", "quartz_pseudo", "social_i_qa_pseudo", "cosmos_qa_pseudo", "quarel_pseudo"],
    "EXQA": ["adversarial_qa", "quoref", "ropes", "duorc_self", "duorc_para"],
    "EXQA_PSEUDO": ["adversarial_qa_pseudo", "quoref_pseudo", "ropes_pseudo"],
    "CBQA": ["hotpot_qa_distractor", "hotpot_qa_fullwiki", "wiki_qa"],
    "CBQA_PSEUDO": ["wiki_qa_pseudo"],
    "SENT": ["yelp_polarity", "rotten_tomatoes", "imdb", "app_reviews", "amazon_polarity"],
    "SENT_PSEUDO": ["yelp_polarity_pseudo", "rotten_tomatoes_pseudo", "imdb_pseudo"],
    "TC": ["ag_news", "dbpedia_14", "trec"],
    "TC_PSEUDO": ["ag_news_pseudo", "dbpedia_14_pseudo"],
    "S2T": ["wiki_bio", "common_gen"],
    "S2T_PSEUDO": ["common_gen_pseudo"],
    "SUM": ["xsum", "gigaword", "multi_news", "samsum", "cnn_dailymail"],
    "SUM_PSEUDO": ["xsum_pseudo", "gigaword_pseudo", "cnn_dailymail_pseudo"],
    "PARA": ["mrpc", "qqp", "paws_labeled_final"],
    "PARA_PSEUDO": ["mrpc_pseudo", "qqp_pseudo", "paws_labeled_final_pseudo"],
    "SC": ["copa", "hellaswag", "story_cloze_2016"],
    "NLI": ["rte", "cb", "anli_r1", "anli_r2", "anli_r3"],
    "COREF": ["wsc", "winorande_xl", "winogrande_debiased"],
    "WIC": ["wic"],
    "DATA_0.1": ["sciq", "duorc_self", "hotpot_qa_fullwiki", "app_reviews", "trec", "wiki_bio", "multi_news", "qqp"],
    "DATA_0.4": ["qasc", "quarel", "wiki_hop", "quoref", "imdb", "xsum"],
    "DATA_0.7": ["wiqa", "quartz", "dream", "ropes", "duorc_para", "hotpot_qa_distractor", "amazon_polarity", "rotten_tomatoes", "ag_news", "samsum", "cnn_dailymail", "mrpc"],
    "DATA_1.0": ["cosmos_qa", "social_i_qa", "quail", "adversarial_qa", "wiki_qa", "yelp_polarity", "dbpedia_14", "common_gen", "gigaword", "paws_labeled_final"],
    "DATA_0.4_PSEUDO": ["quarel_pseudo", "quoref_pseudo", "imdb_pseudo", "xsum_pseudo"],
    "DATA_0.7_PSEUDO": ["quartz_pseudo", "ropes_pseudo", "rotten_tomatoes_pseudo", "ag_news_pseudo", "cnn_dailymail_pseudo", "mrpc_pseudo"],
    "DATA_1.0_PSEUDO": ["cosmos_qa_pseudo", "social_i_qa_pseudo", "quail_pseudo", "adversarial_qa_pseudo", "wiki_qa_pseudo", "yelp_polarity_pseudo", "dbpedia_14_pseudo", "common_gen_pseudo", "gigaword_pseudo", "paws_labeled_final_pseudo"],
}

DATA_NO_VALID = [
    ("hotpot_qa_distractor", None),
    ("hotpot_qa_fullwiki", None),
    ("paws_labeled_final", None),
    ("paws_labeled_final_pseudo", None),
    ("adversarial_qa", None),
    ("adversarial_qa_pseudo", None),
    ("duorc_ParaphraseRC", None),
    ("dream", None),
    ("amazon_polarity", None),
    ("app_reviews", None),
    ("app_reviews_pseudo", None),
    ("imdb", None),
    ("imdb_pseudo", None),
    ("wiki_bio", None),
    ("gigaword", None),
    ("gigaword_pseudo", None),
    ("multi_news", None),
    ("samsum", None),
    ("dbpedia_14", None),
    ("dbpedia_14_pseudo", None),
    ("trec", None),
]

DATA_NO_EVAL = [
    ("story_cloze_2016", "Generate Ending"),
    ("story_gen", "Answer Given options"),
    ("story_gen", "Choose Story Ending"),
    ("story_gen", "Movie What Happens Next"),
    ("story_gen", "Novel Correct Ending"),
    ("story_gen", "Story Continuation and Options"),
    ("squad_qg", "answer_given_context_and_question"),
    ("squad_qg", "answer_question_given_context"),
    ("squad_qg", "answer_the_question"),
    ("squad_qg", "given_context_answer_question_variation"),
    ("hellaswag", "Open-ended completion"),
    ("hellaswag", "Open-ended start"),
    ("hellaswag", "Topic of the context"),
    ("hellaswag", "Reversed appropriate continuation - Yes or No"),
    ("hellaswag", "Appropriate continuation - Yes or No"),
    ("hellaswag", "Topic without the ending answer"),
]

DATA_NO_TRAIN = {
    ("wiki_qa_pseudo", "Topic Prediction - Answer Only"),
    ("wiki_qa_pseudo", "Topic Prediction - Question Only"),
    ("wiki_qa_pseudo", "Topic Prediction - Question and Answer Pair"),
    ("app_reviews", "convert_to_star_rating")
}

DATA_EVAL_GEN = {
    ("story_gen", "Generate Ending")
}

DATA_CONFIG = {
    "commonsense_qa": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "commonsense_qa/cache")
    },
    "dream": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "dream/cache")
    },
    "quail": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quail/cache")
    },
    "quail_pseudo": {
        "name": ["quail", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quail/pseudo")
    },
    "quartz": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quartz/cache")
    },
    "quartz_pseudo": {
        "name": ["quartz", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quartz/pseudo")
    },
    "social_i_qa": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "social_i_qa/cache")
    },
    "social_i_qa_pseudo": {
        "name": ["social_i_qa", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "social_i_qa/pseudo")
    },
    "social_i_qa_pseudo": {
        "name": ["social_i_qa", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "social_i_qa/pseudo")
    },
    "wiqa": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wiqa/cache")
    },
    "cosmos_qa": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "cosmos_qa/cache")
    },
    "cosmos_qa_pseudo": {
        "name": ["cosmos_qa", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "cosmos_qa/pseudo")
    },
    "qasc": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "qasc/cache")
    },
    "quarel": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quarel/cache")
    },
    "quarel_pseudo": {
        "name": ["quarel", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quarel/pseudo")
    },
    "sciq": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "sciq/cache")
    },
    "sciq_sciq": {
        "name": ["sciq", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "sciq/from_sciq")
    },
    "wiki_hop": {
        "name": ["wiki_hop", "original"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wiki_hop/cache/"),
        "split": ["train", "validation"],
    },
    "adversarial_qa": {
        "name": ["adversarial_qa", "adversarialQA"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "adversarial_qa/cache/adversarialQA")
    },
    "adversarial_qa_pseudo": {
        "name": ["adversarial_qa", "adversarialQA"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "adversarial_qa/pseudo/adversarialQA")
    },
    "quoref": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quoref/cache")
    },
    "quoref_pseudo": {
        "name": ["quoref", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quoref/pseudo")
    },
    "quoref_duorc": {
        "name": ["quoref", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "quoref/from_duorc")
    },
    "ropes": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "ropes/cache")
    },
    "ropes_pseudo": {
        "name": ["ropes", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "ropes/pseudo")
    },
    "duorc_self": {
        "name": ["duorc", "SelfRC"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "duorc/cache/SelfRC")
    },
    "duorc_para": {
        "name": ["duorc", "ParaphraseRC"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "duorc/cache/ParaphraseRC")
    },
    "duorc_para_duorc": {
        "name": ["duorc", "ParaphraseRC"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "duorc/from_duorc/ParaphraseRC")
    },
    "hotpot_qa_distractor": {
        "name": ["hotpot_qa", "distractor"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "hotpot_qa/cache/distractor")
    },
    "hotpot_qa_fullwiki": {
        "name": ["hotpot_qa", "fullwiki"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "hotpot_qa/cache/fullwiki")
    },
    "hotpot_qa_fullwiki_hotpot": {
        "name": ["hotpot_qa", "fullwiki"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "hotpot_qa/from_hotpot_qa/fullwiki")
    },
    "wiki_qa": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wiki_qa/cache")
    },
    "wiki_qa_pseudo": {
        "name": ["wiki_qa", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wiki_qa/pseudo")
    },
    "amazon_polarity": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "amazon_polarity/cache")
    },
    "app_reviews": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "app_reviews/cache")
    },
    "app_reviews_pseudo": {
        "name": ["app_reviews", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "app_reviews/pseudo")
    },
    "imdb": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "imdb/cache")
    },
    "imdb_pseudo": {
        "name": ["imdb", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "imdb/pseudo")
    },
    "rotten_tomatoes": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "rotten_tomatoes/cache")
    },
    "rotten_tomatoes_pseudo": {
        "name": ["rotten_tomatoes", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "rotten_tomatoes/pseudo")
    },
    "yelp_polarity": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "yelp_polarity/cache")
    },
    "yelp_polarity_pseudo": {
        "name": ["yelp_polarity", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "yelp_polarity/pseudo")
    },
    "ag_news": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "ag_news/cache")
    },
    "ag_news_pseudo": {
        "name": ["ag_news", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "ag_news/pseudo")
    },
    "dbpedia_14": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "dbpedia_14/cache")
    },
    "dbpedia_14_pseudo": {
        "name": ["dbpedia_my", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "dbpedia_14/pseudo")
    },
    "trec": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "trec/cache")
    },
    "common_gen": {
        "type": "gen",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "common_gen/cache")
    },
    "common_gen_pseudo": {
        "name": ["common_gen", None],
        "type": "gen",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "common_gen/pseudo")
    },
    "wiki_bio": {
        "type": "gen",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wiki_bio/cache")
    },
    "cnn_dailymail": {
        "name": ["cnn_dailymail", "3.0.0"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "cnn_dailymail/cache/3.0.0")
    },
    "cnn_dailymail_pseudo": {
        "name": ["cnn_dailymail", "3.0.0"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "cnn_dailymail/pseudo/3.0.0")
    },
    "gigaword": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "gigaword/cache")
    },
    "gigaword_pseudo": {
        "name": ["gigaword", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "gigaword/pseudo")
    },
    "multi_news": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "multi_news/cache")
    },
    "samsum": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "samsum/cache")
    },
    "xsum": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "xsum/cache")
    },
    "xsum_pseudo": {
        "name": ["xsum", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "xsum/pseudo")
    },
    "mrpc": {
        "name": ["glue", "mrpc"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "mrpc/cache")
    },
    "mrpc_pseudo": {
        "name": ["glue", "mrpc"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "mrpc/pseudo")
    },
    "paws_labeled_final": {
        "name": ["paws", "labeled_final"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "paws/cache/labeled_final")
    },
    "paws_labeled_final_pseudo": {
        "name": ["paws", "labeled_final"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "paws/pseudo/labeled_final")
    },
    "paws_labeled_final_qqp": {
        "name": ["paws", "labeled_final"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "paws/from_qqp/labeled_final")
    },
    "qqp": {
        "name": ["glue", "qqp"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "qqp/cache")
    },
    "qqp_pseudo": {
        "name": ["glue", "qqp"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "qqp/pseudo")
    },
    "copa": {
        "name": ["super_glue", "copa"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "copa/cache")
    },
    "hellaswag": {
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "hellaswag/cache")
    },
    "story_cloze_2016": {
        "name": ["story_cloze", "2016"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "story_cloze/cache/2016"),
        "split": ["validation"]
    },
    "story_gen": {
        "name": ["story_cloze", "2016"],
        "type": "gen",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "story_cloze/cache/2016"),
        "split": ["validation"]
    },
    "squad_qg": {
        "name": ["squad", None],
        "type": "gen",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "squad/cache/"),
        "split": ["validation"]
    },
    "anli_r1": {
        "name": ["anli", None],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "anli/cache/r1")
    },
    "cb": {
        "name": ["super_glue", "cb"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "cb/cache")
    },
    "rte": {
        "name": ["super_glue", "rte"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "rte/cache")
    },
    "wsc": {
        "name": ["super_glue", "wsc.fixed"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wsc_balance/cache")
    },
    "winogrande_xl": {
        "name": ["winogrande", "winogrande_xl"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "winogrande/cache/winogrande_xl")
    },
    "winogrande_debiased": {
        "name": ["winogrande", "winogrande_debiased"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "winogrande/cache/winogrande_debiased")
    },
    "wic": {
        "name": ["super_glue", "wic"],
        "type": "rank",
        "do_cache": True,
        "data_dir": os.path.join(BASE_DATA_DIR, "wic/cache")
    },
    "nsg": {
        "type": "gen",
        "selfsup": True,
        "do_cache": True,
        "flan_sample_max": 30000,
        "data_dir": os.path.join(BASE_DATA_DIR, "selfsup/merge/nsg"),
        "metric": None
    },
    "mwp": {
        "type": "gen",
        "selfsup": True,
        "do_cache": True,
        "flan_sample_max": 30000,
        "data_dir": os.path.join(BASE_DATA_DIR, "selfsup/merge/mwp"),
        "metric": None
    },
    "lpp_gen": {
        "type": "gen",
        "selfsup": True,
        "do_cache": True,
        "flan_sample_max": 30000,
        "data_dir": os.path.join(BASE_DATA_DIR, "selfsup/merge/lpp_gen"),
        "metric": None
    },
    "lpp_cls": {
        "type": "rank",
        "selfsup": True,
        "do_cache": True,
        "flan_sample_max": 30000,
        "data_dir": os.path.join(BASE_DATA_DIR, "selfsup/merge/lpp_cls"),
        "metric": None
    },
}