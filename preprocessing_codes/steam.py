import pandas as pd
import os

# === CONFIG ===
input_file = "/Users/nitinbisht/Downloads/taobao/steam.txt"
output_file = "/Users/nitinbisht/Downloads/taobao/output/steam_final.csv"
min_interactions = 100
max_users = 10000

# === Load file (whitespace-separated) ===
df = pd.read_csv(input_file, sep=r'\s+', header=None, names=["user_id", "item_id"])

# === Filter users with at least `min_interactions` ===
user_counts = df['user_id'].value_counts()
eligible_users = user_counts[user_counts >= min_interactions].index[:max_users]

df = df[df['user_id'].isin(eligible_users)].copy()
print(f"✅ Users with ≥{min_interactions} interactions: {len(eligible_users):,}")
print(f"📊 Remaining rows after filtering: {len(df):,}")

# === Sort by user only (NOT item) to preserve sequence ===
df['original_order'] = range(len(df))  # preserve interaction order
df = df.sort_values(by=["user_id", "original_order"]).drop(columns=["original_order"]).reset_index(drop=True)

# === Assign synthetic timestamps per user ===
print("⏱️ Assigning synthetic timestamps...")

timestamps = []
base_time = 1_500_000_000  # Starting UNIX time

for _, group in df.groupby("user_id"):
    n = len(group)
    group_ts = list(range(base_time, base_time + n))
    timestamps.extend(group_ts)
    base_time += n

df["timestamp"] = timestamps

# === Reindex users and items ===
user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
item2id = {it: i for i, it in enumerate(df['item_id'].unique())}

df["user_idx"] = df["user_id"].map(user2id)
df["item_idx"] = df["item_id"].map(item2id)

# === Final DataFrame ===
final_df = df[["user_idx", "item_idx", "timestamp"]]

# === Save output ===
os.makedirs(os.path.dirname(output_file), exist_ok=True)
final_df.to_csv(output_file, index=False)

# === Summary ===
print("✅ Done! Steam dataset processed with filters.")
print(f"📁 Output: {output_file}")
print(f"👤 Unique users: {final_df['user_idx'].nunique():,}")
print(f"🎮 Unique items: {final_df['item_idx'].nunique():,}")
print(f"🧾 Total interactions: {len(final_df):,}")
