import torch
import json
import time
import sys
import random
from tqdm import tqdm

from transformers import MarianMTModel, MarianTokenizer


def translate(texts, model, tokenizer, language="fr", device="cpu"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    s = time.time()
    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors="pt")
    for k in encoded:
        encoded[k] = encoded[k].to(device)
    t = time.time()
    # Generate translation using model
    translated = model.generate(**encoded)
    e = time.time()
    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    d = time.time()

    return translated_texts


def back_translate(texts, en_model, en_tokenizer, target_model, target_tokenizer, source_lang="en", target_lang="fr", device="cpu"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer,
                         language=target_lang, device=device)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer,
                                      language=source_lang, device=device)

    return back_translated_texts


def main():
    device = torch.cuda.current_device()
    random.seed(20)

    target_model_name = "/home/yourname/checkpoints/mt/en-ROMANCE/"
    target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
    target_model = MarianMTModel.from_pretrained(target_model_name).to(device)

    en_model_name = "/home/yourname/checkpoints/mt/ROMANCE-en/"
    en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
    en_model = MarianMTModel.from_pretrained(en_model_name).to(device)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        lines = f.readlines()

    data = [json.loads(line) for line in lines]

    data = data

    pos_data = [d["pos"] for d in data]
    neg_data = [d["neg"] for d in data]
    
    batch_size = 32
    
    all_aug_pos_texts = []    
    for i in tqdm(range(0, len(pos_data), batch_size)):
        batch = pos_data[i:i+batch_size]
        aug_texts = back_translate(batch, en_model, en_tokenizer, target_model, target_tokenizer, source_lang="en", target_lang="es", device=device)
        all_aug_pos_texts.extend(aug_texts)

    all_aug_neg_texts = []
    for i in tqdm(range(0, len(neg_data), batch_size)):
        batch = neg_data[i:i+batch_size]
        aug_texts = back_translate(batch, en_model, en_tokenizer, target_model, target_tokenizer, source_lang="en", target_lang="es", device=device)
        all_aug_neg_texts.extend(aug_texts)

    with open(output_path, "w") as f:
        for d, pos_d in zip(pos_data, all_aug_pos_texts):
            sents = [d, pos_d]
            random.shuffle(sents)
            f.write(json.dumps({"sentence1": sents[0], "sentence2": sents[1], "label": 1}) + "\n")
        
        for d, neg_d in zip(pos_data, all_aug_neg_texts):
            sents = [d, neg_d]
            random.shuffle(sents)
            f.write(json.dumps({"sentence1": sents[0], "sentence2": sents[1], "label": 0}) + "\n")


if __name__ == "__main__":
    main()
