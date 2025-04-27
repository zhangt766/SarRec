# dump_lgcn_emb.py

import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# —— 1. 确保能 import 到你的项目 —— 
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# —— 2. 从 TF checkpoint 里直接载入变量 —— 
from tensorflow.python.training import checkpoint_utils

# checkpoint 前缀（去掉 .index/.data 后缀）
CKPT_DIR  = os.path.join(PROJECT_ROOT,
    "weights/Movielens/LightGCN/64-64-64/l0.01_r1e-05")
CKPT_PATH = tf.train.latest_checkpoint(CKPT_DIR)
if CKPT_PATH is None:
    raise FileNotFoundError(f"在 {CKPT_DIR} 下找不到 checkpoint")
print("✅ 找到 checkpoint：", CKPT_PATH)

# 3. 载入 item_embedding
#    名称要跟 LightGCN._init_weights() 中的 name 一致
item_emb = checkpoint_utils.load_variable(CKPT_PATH, "item_embedding")
print("✅ 载入 item_embedding，shape =", item_emb.shape)

# 4. 保存为 .npy
out_dir = os.path.join(PROJECT_ROOT, "data/ml-1m/saved_embed")
os.makedirs(out_dir, exist_ok=True)
emb_path = os.path.join(out_dir, "lightgcn_item_emb.npy")
np.save(emb_path, item_emb)
print("✅ 保存 embedding 到", emb_path)

# 5. 重建 idx -> 原始 itemID 映射
train_f = os.path.join(PROJECT_ROOT, "Data/Movielens/train.txt")
test_f  = os.path.join(PROJECT_ROOT, "Data/Movielens/test.txt")

item_set = set()
for fn in (train_f, test_f):
    with open(fn, "r") as f:
        for line in f:
            parts = line.strip().split()
            # 第一列通常是 user，后面都是 itemID
            for tok in parts[1:]:
                item_set.add(int(tok))

sorted_items = sorted(item_set)
id2item = { str(idx): str(item_id)
            for idx, item_id in enumerate(sorted_items) }

map_path = os.path.join(out_dir, "lightgcn_id2item.json")
with open(map_path, "w", encoding="utf-8") as f:
    json.dump(id2item, f, indent=2, ensure_ascii=False)
print(f"✅ 保存映射到 {map_path}，共 {len(id2item)} 条")
