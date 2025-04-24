import pandas as pd
from collections import Counter
from tqdm import tqdm

filename = "Gowalla.txt"
output_file = "gowalla_cleaned.csv"
min_interactions = 70
sample_users = 2500  # Set to None to keep all eligible users

data = []
print("Reading file...")
with open(filename, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Lines processed"):
        parts = line.strip().split('\t')
        if len(parts) != 5:
            continue
        user_id, timestamp, _, _, location_id = parts
        data.append((user_id, timestamp, location_id))

df = pd.DataFrame(data, columns=["user_id", "timestamp", "item_id"])
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

user_counts = df['user_id'].value_counts()
eligible_users = user_counts[user_counts >= min_interactions].index
df = df[df['user_id'].isin(eligible_users)].copy()

if sample_users is not None:
    sampled_users = df['user_id'].drop_duplicates().sample(n=sample_users, random_state=42)
    df = df[df['user_id'].isin(sampled_users)].copy()

user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
item2id = {i: j for j, i in enumerate(df['item_id'].unique())}
df['user_idx'] = df['user_id'].map(user2id)
df['item_idx'] = df['item_id'].map(item2id)

final_df = df[['timestamp', 'user_idx', 'item_idx']].sort_values(by=['user_idx', 'timestamp']).reset_index(drop=True)
final_df.to_csv(output_file, index=False)

print("Saved to:", output_file)
print("Final stats:")
print("  Interactions:", len(df))
print("  Users:", df['user_idx'].nunique())
print("  Items:", df['item_idx'].nunique())
print("  Avg interactions/user:", round(len(df) / df['user_idx'].nunique(), 2))
