# data_utils.py
import json
import numpy as np
from datasets import load_dataset

def load_data(data_path, train_size, val_size, K, train_type, temp_type, emb_type, idx2item):
    item2idx = {int(v): int(k) for k, v in idx2item.items()}
    
    DATA_PATH = {
        "train": f"{data_path}/train/train_{K}_{train_type}_{emb_type}_sampled.json",
        "test": f"{data_path}/test/test_{K}_{temp_type}_{emb_type}_sampled.json"
    }

    data = load_dataset("json", data_files=DATA_PATH)
    data["train"] = data["train"].select(range(train_size))
    data["test"] = data["test"].select(range(val_size))

    session_data = []
    user_ids = []
    with open(DATA_PATH["train"], 'r', encoding='utf-8') as fin:
        for line in fin:
            rec = json.loads(line.strip())
            raw_hist = rec.get('history', [])
            raw_nxt = rec.get('label')
            hist_idx = [item2idx[mid] for mid in raw_hist]
            nxt_idx = item2idx[raw_nxt]
            session_data.append((hist_idx, nxt_idx))
            user_ids.append(str(rec["user_id"]))

    return data, session_data, user_ids

def load_embeddings(item_emb_path):
    item_emb_np = np.load(item_emb_path)
    item_emb = torch.from_numpy(item_emb_np).float().cuda()
    num_items = item_emb.size(0)
    emb_dim = item_emb_np.shape[1]
    return item_emb, num_items, emb_dim
