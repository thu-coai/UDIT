from tokenization_t5 import EncDecTokenizer

tokenizer = EncDecTokenizer("/home/yourname/UDIT/vocab_en/spiece.model")

with open("/home/yourname/UDIT/vocab_en/vocab.txt", "w") as f:
    for i in range(32100):
        f.write(tokenizer.convert_id_to_token(i) + "\n")
    
print(tokenizer.encode("Maybe"))