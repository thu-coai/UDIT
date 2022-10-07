import pickle

def get_sizes(data):
    enc_sizes = [len(s["enc_input_ids"]) for s in data]
    dec_sizes = [len(s["dec_input_ids"]) for s in data]
    cand_sizes = [sum([len(c) for c in s["cands"]]) if s["cands"] is not None else 0 for s in data]
    return enc_sizes, dec_sizes, cand_sizes


train_num = 100000
valid_num = 1000

for d in ["lpp_cls"]:
    with open("/home/yourname/UDIT/self_sup_data/selfsup/merge/{}/cache_all_-1.pkl".format(d), "rb") as f:
        all_samples = pickle.load(f)

    print(len(all_samples))
    with open("/home/yourname/UDIT/self_sup_data/selfsup/merge/{}/cache_train_{}.pkl".format(d, train_num), "wb") as f:
        train_data = all_samples[:train_num]
        for i in range(len(train_data)):
            train_data[i]["cands"] = None
            train_data[i]["options"] = None
            train_data[i]["idxs"][2] = i
        enc_sizes, dec_sizes, cand_sizes = get_sizes(train_data)
        pickle.dump((train_data, enc_sizes, dec_sizes, cand_sizes), f)

    with open("/home/yourname/UDIT/self_sup_data/selfsup/merge/{}/cache_validation_{}.pkl".format(d, valid_num), "wb") as f:
        valid_data = all_samples[train_num:train_num + valid_num]
        for i, d in enumerate(valid_data):
            d["idxs"][2] = i
        enc_sizes, dec_sizes, cand_sizes = get_sizes(valid_data)
        pickle.dump((valid_data, enc_sizes, dec_sizes, cand_sizes), f)
