import pandas as pd
from collections import Counter
from tqdm import tqdm
import os
from random import randint

# Config
input_file = "/Users/nitinbisht/Downloads/taobao/goodreads_interactions.csv"
output_file = "/Users/nitinbisht/Downloads/taobao/output/good_reads_final.csv"
chunk_size = 100_000
min_interactions = 100
max_users = 1800

# Pass 1: Count interactions and collect eligible users
print("Counting user interactions (stopping early at 10,000 users)...")
user_counts = Counter()
eligible_set = set()

# Detect user column from first chunk
first_chunk = pd.read_csv(input_file, nrows=1)
user_col = next((col for col in first_chunk.columns if 'user' in col.lower()), None)
if not user_col:
    raise ValueError("Could not find a user column.")

for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    counts = chunk[user_col].value_counts()
    for user, count in counts.items():
        user_counts[user] += count
        if user_counts[user] >= min_interactions:
            eligible_set.add(user)
    if len(eligible_set) >= max_users:
        print("Reached 10,000 eligible users. Stopping early.")
        break

eligible_users = list(eligible_set)[:max_users]
eligible_set = set(eligible_users)
print(f"Total eligible users: {len(eligible_set):,}")

# Pass 2: Filter rows with eligible users
item_col = next((col for col in first_chunk.columns if any(x in col.lower() for x in ['item', 'book', 'bid'])), None)
if not item_col:
    raise ValueError("Could not find an item/book column.")

filtered_rows = []
print("Filtering rows for eligible users...")
for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size), desc="Processing chunks"):
    chunk = chunk[[user_col, item_col]].dropna()
    chunk = chunk[chunk[user_col].isin(eligible_set)]
    filtered_rows.append(chunk)

df = pd.concat(filtered_rows, ignore_index=True)
df.rename(columns={user_col: "user_id", item_col: "item_id"}, inplace=True)

# Trim to 500–1000 interactions per user
print("Trimming to 500–1000 interactions per user...")
trimmed_rows = []
for user_id, group in tqdm(df.groupby("user_id"), desc="Trimming users", unit="user"):
    group = group.sort_index()
    n = len(group)
    if n < 500:
        continue
    max_len = min(800, n)
    keep_n = randint(500, max_len)
    trimmed = group.iloc[:keep_n]
    trimmed_rows.append(trimmed)

df = pd.concat(trimmed_rows).reset_index(drop=True)

# Simulate timestamps
print("Simulating timestamps...")
df["original_order"] = range(len(df))
df = df.sort_values(by=["user_id", "original_order"]).drop(columns=["original_order"]).reset_index(drop=True)

timestamps = []
base_time = 1_500_000_000
for _, group in tqdm(df.groupby("user_id"), desc="Generating timestamps", unit="user"):
    n = len(group)
    timestamps.extend(range(base_time, base_time + n))
    base_time += n

df["timestamp"] = timestamps

# Reindex user and item IDs
user2id = {u: i for i, u in enumerate(df["user_id"].unique())}
item2id = {i: j for j, i in enumerate(df["item_id"].unique())}

df["user_idx"] = df["user_id"].map(user2id)
df["item_idx"] = df["item_id"].map(item2id)

final_df = df[["user_idx", "item_idx", "timestamp"]]

# Save output
os.makedirs(os.path.dirname(output_file), exist_ok=True)
final_df.to_csv(output_file, index=False)

# Summary
print("Processing complete.")
print(f"File saved to: {output_file}")
print(f"Users: {final_df['user_idx'].nunique():,}")
print(f"Items: {final_df['item_idx'].nunique():,}")
print(f"Total interactions: {len(final_df):,}")
