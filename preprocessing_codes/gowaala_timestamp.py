import pandas as pd
import os

# === Paths ===
input_file = "/Users/nitinbisht/Downloads/taobao/gowalla_final.csv"
output_file = "/Users/nitinbisht/Downloads/taobao/output/gowalla_final.csv"

# === Load the file ===
df = pd.read_csv(input_file)

# === Convert timestamp to UNIX seconds ===
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['timestamp'] = df['timestamp'].astype('int64') // 10**9  # convert to seconds

# === Save cleaned file ===
df.to_csv(output_file, index=False)

# === Summary ===
print("✅ Timestamps converted and saved!")
print(f"📁 New file: {output_file}")
print(f"🔢 Total rows: {len(df):,}")
print(f"📅 Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")
