import pandas as pd
import numpy as np
from collections import defaultdict
import os

input_file = "/Users/nitinbisht/Downloads/taobao/UserBehavior.csv"
output_folder = "/Users/nitinbisht/Downloads/taobao/output"
output_file = os.path.join(output_folder, "taobao_final.csv")

os.makedirs(output_folder, exist_ok=True)

SECONDS_IN_DAY = 86400
min_ts = 1511544070
MAX_DAYS = 120
cutoff = min_ts + MAX_DAYS * SECONDS_IN_DAY

user_data = defaultdict(list)

with open(input_file, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split(',')
        if len(parts) != 5:
            continue
        user, item, cat, action, ts = parts
        if action.strip() != 'pv':
            continue
        try:
            ts = int(ts.strip())
            if not (min_ts <= ts < cutoff):
                continue
        except ValueError:
            continue
        user_data[user].append((ts, item))
        if i % 10_000_000 == 0 and i > 0:
            print(f"Processed {i:,} lines...")

print(f"Collected data for {len(user_data)} users.")

qualified_users = {
    user: events for user, events in user_data.items() if len(events) >= 100
}
print(f"Users with ≥100 interactions: {len(qualified_users)}")

user_spans = {
    user: max(ts for ts, _ in events) - min(ts for ts, _ in events)
    for user, events in qualified_users.items()
}

top_users_by_span = sorted(
    user_spans, key=user_spans.get, reverse=True
)[:1500]

for u in top_users_by_span[:5]:
    span_days = user_spans[u] / SECONDS_IN_DAY
    print(f"User {u}: {span_days:.1f} days")

rows = []
for user in top_users_by_span:
    for ts, item in sorted(qualified_users[user]):
        rows.append((user, item, ts))

df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])

print("Dataset summary:")
print(f"Total interactions: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")
print(f"Avg interactions per user: {round(len(df)/df['user_id'].nunique(), 2)}")

df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
