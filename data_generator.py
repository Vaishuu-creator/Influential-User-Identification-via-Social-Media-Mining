"""
data_generator.py  -  Influential User Identification via Social Media Mining
==============================================================================
Generates a synthetic Twitter-like dataset that mirrors the structure of:

  Kaggle Dataset : "Twitter User Data" (ego-networks / follower graphs)
  URL            : https://www.kaggle.com/datasets/hwassner/TwitterFriends

Produces inside ./data/:
  users.csv   - 500 user profiles with engagement attributes
  edges.csv   - 4000 directed follower -> followee relationships
  tweets.csv  - 3000 tweet records with engagement metrics
"""

import numpy as np
import pandas as pd
import random
import os

random.seed(42)
np.random.seed(42)

N_USERS  = 500
N_EDGES  = 4000
N_TWEETS = 3000
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

TOPICS = ["AI", "Politics", "Sports", "Tech", "Music",
          "Gaming", "Finance", "Health", "Travel", "Fashion"]

TEXTS = [
    "Just published a new post on {topic}!",
    "Excited about the latest developments in {topic}.",
    "What do you think about {topic}? Let me know.",
    "Big news in the {topic} world today!",
    "Hot take on {topic} -- long thread incoming.",
    "New research on {topic} just dropped -- fascinating.",
    "Cannot believe what is happening in {topic}.",
    "Sharing my {topic} predictions for this year.",
    "This {topic} trend is changing everything.",
    "My experience with {topic} so far this month.",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. USERS
# ─────────────────────────────────────────────────────────────────────────────
user_ids = list(range(1, N_USERS + 1))
followers = np.random.zipf(1.8, N_USERS) * 10   # power-law distribution

users_df = pd.DataFrame({
    "user_id":          user_ids,
    "screen_name":      [f"user_{i:04d}" for i in user_ids],
    "verified":         np.random.choice([0, 1], N_USERS, p=[0.92, 0.08]),
    "followers_count":  followers,
    "following_count":  np.random.randint(50, 2000, N_USERS),
    "tweet_count":      np.random.negative_binomial(5, 0.1, N_USERS),
    "listed_count":     np.random.negative_binomial(2, 0.3, N_USERS),
    "account_age_days": np.random.randint(30, 4000, N_USERS),
    "favourite_topic":  np.random.choice(TOPICS, N_USERS),
})
users_df.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)
print(f"[+] users.csv  -- {len(users_df)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDGES  (directed follower -> followee, preferential attachment)
# ─────────────────────────────────────────────────────────────────────────────
weights = followers.astype(float)
weights /= weights.sum()

edges = set()
attempts = 0
while len(edges) < N_EDGES and attempts < N_EDGES * 20:
    fr = random.randint(1, N_USERS)
    to = int(np.random.choice(user_ids, p=weights))
    if fr != to:
        edges.add((fr, to))
    attempts += 1

pd.DataFrame(list(edges), columns=["follower_id", "followee_id"]).to_csv(
    os.path.join(OUT_DIR, "edges.csv"), index=False
)
print(f"[+] edges.csv  -- {len(edges)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TWEETS
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for tid in range(1, N_TWEETS + 1):
    uid   = random.randint(1, N_USERS)
    topic = random.choice(TOPICS)
    rows.append({
        "tweet_id":      tid,
        "user_id":       uid,
        "topic":         topic,
        "text":          random.choice(TEXTS).format(topic=topic),
        "retweet_count": int(np.random.negative_binomial(1, 0.2)),
        "like_count":    int(np.random.negative_binomial(2, 0.15)),
        "reply_count":   int(np.random.negative_binomial(1, 0.4)),
        "hashtag_count": random.randint(0, 5),
        "mention_count": random.randint(0, 3),
    })

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "tweets.csv"), index=False)
print(f"[+] tweets.csv -- {N_TWEETS} rows")
print("\nData generation complete. Files are in ./data/")