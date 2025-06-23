# Interaction_emb.py

import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tensorflow.python.training import checkpoint_utils


CKPT_DIR  = os.path.join(PROJECT_ROOT,
    "weights/Movielens/LightGCN/64-64-64/l0.01_r1e-05")
CKPT_PATH = tf.train.latest_checkpoint(CKPT_DIR)
if CKPT_PATH is None:
    raise FileNotFoundError(f"在 {CKPT_DIR} 下找不到 checkpoint")
print("✅ find checkpoint：", CKPT_PATH)

item_emb = checkpoint_utils.load_variable(CKPT_PATH, "item_embedding")
print("✅ loading item_embedding，shape =", item_emb.shape)

out_dir = os.path.join(PROJECT_ROOT, "data/ml-1m/saved_embed")
os.makedirs(out_dir, exist_ok=True)
emb_path = os.path.join(out_dir, "lightgcn_item_emb.npy")
np.save(emb_path, item_emb)
print("✅ save embedding ", emb_path)


train_f = os.path.join(PROJECT_ROOT, "Data/Movielens/train.txt")
test_f  = os.path.join(PROJECT_ROOT, "Data/Movielens/test.txt")

item_set = set()
for fn in (train_f, test_f):
    with open(fn, "r") as f:
        for line in f:
            parts = line.strip().split()
            for tok in parts[1:]:
                item_set.add(int(tok))

sorted_items = sorted(item_set)
id2item = { str(idx): str(item_id)
            for idx, item_id in enumerate(sorted_items) }

map_path = os.path.join(out_dir, "lightgcn_id2item.json")
with open(map_path, "w", encoding="utf-8") as f:
    json.dump(id2item, f, indent=2, ensure_ascii=False)
print(f"{map_path}， {len(id2item)}")
