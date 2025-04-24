import pandas as pd
from collections import Counter
from tqdm import tqdm
import os
from random import randint

input_file = "/Users/nitinbisht/Downloads/taobao/lastfm.tsv"
shared_output_folder = "/Users/nitinbisht/Downloads/taobao/output"
output_file = os.path.join(shared_output_folder, "lastfm_processed_500users.csv")

time_format = "%Y-%m-%dT%H:%M:%SZ"
sample_users = 600
min_interactions = 200

os.makedirs(shared_output_folder, exist_ok=True)

delimiters = ['\t', ',', ';', '|', ' ']
with open(input_file, 'r', encoding='utf-8') as f:
    sample_lines = [f.readline().strip() for _ in range(10)]

best_delim = None
best_score = 0

for delim in delimiters:
    split_counts = [len(line.split(delim)) for line in sample_lines]
    count_freq = Counter(split_counts)
    most_common_count, freq = count_freq.most_common(1)[0]
    score = freq * most_common_count
    if score > best_score:
        best_score = score
        best_delim = delim

print(f"Detected delimiter: {repr(best_delim)}\n")

data = []
count = 0
print("Reading file...")

with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Lines processed", unit="lines"):
        parts = line.strip().split(best_delim)
        if len(parts) >= 3:
            data.append(parts[:3])
            count += 1
            if count % 500000 == 0:
                print(f"{count:,} rows read...")

print(f"Total rows loaded: {count:,}\n")

df = pd.DataFrame(data, columns=["user_id", "timestamp", "item_id"])
df['timestamp'] = pd.to_datetime(df['timestamp'], format=time_format, errors='coerce')
df = df.dropna(subset=['timestamp'])
df['timestamp'] = df['timestamp'].astype('int64') // 10**9
df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

user_counts = df['user_id'].value_counts()
eligible_users = user_counts[user_counts >= min_interactions].index

if len(eligible_users) < sample_users:
    raise ValueError(f"Only {len(eligible_users)} users have ≥{min_interactions} interactions. Reduce sample size.")

top_users = eligible_users.to_series().sample(n=sample_users, random_state=42)
df = df[df['user_id'].isin(top_users)].copy()

print("Trimming interactions for each user...")

trimmed_rows = []
for user_id, group in df.groupby("user_id"):
    group = group.sort_values("timestamp")
    n = len(group)
    if n < 200:
        continue
    max_len = min(500, n)
    k = randint(200, max_len)
    start_idx = randint(0, n - k)
    trimmed = group.iloc[start_idx:start_idx + k]
    trimmed_rows.append(trimmed)

df = pd.concat(trimmed_rows).reset_index(drop=True)

user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
item2id = {it: i for i, it in enumerate(df['item_id'].unique())}

df['user_idx'] = df['user_id'].map(user2id)
df['item_idx'] = df['item_id'].map(item2id)
df = df.drop(columns=['user_id', 'item_id'])

df.to_csv(output_file, index=False)

print("Processing complete.")
print(f"Saved file: {output_file}")
print(f"Total interactions: {len(df):,}")
print(f"Unique users: {df['user_idx'].nunique():,}")
print(f"Unique items: {df['item_idx'].nunique():,}")
print(f"Average interactions per user: {round(len(df) / df['user_idx'].nunique(), 2)}")
