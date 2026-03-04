"""
feature_engineering.py  -  Influential User Identification via Social Media Mining
===================================================================================
Computes a rich feature set for each user by combining:
  - Network graph metrics  (degree, PageRank, betweenness, clustering, HITS)
  - Engagement metrics     (avg retweets, likes, replies per tweet)
  - Profile metadata       (follower ratio, account age, verified status)
  - Activity features      (tweet frequency, hashtag and mention usage)

Output:
  data/features.csv  - one row per user, ready for the ML pipeline
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data():
    users  = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    edges  = pd.read_csv(os.path.join(DATA_DIR, "edges.csv"))
    tweets = pd.read_csv(os.path.join(DATA_DIR, "tweets.csv"))
    print(f"  Loaded {len(users)} users | {len(edges)} edges | {len(tweets)} tweets")
    return users, edges, tweets


def build_graph(edges):
    G = nx.from_pandas_edgelist(
        edges, source="follower_id", target="followee_id",
        create_using=nx.DiGraph())
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_network_features(G, user_ids):
    print("  Computing PageRank, HITS, betweenness, clustering...")
    all_ids = list(user_ids)
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pagerank        = nx.pagerank(G, alpha=0.85, max_iter=300)
    hits_hub, hits_auth = nx.hits(G, max_iter=300, normalized=True)
    G_und   = G.to_undirected()
    between = nx.betweenness_centrality(G_und, normalized=True, k=min(150, len(G_und)))
    clust   = nx.clustering(G_und)
    records = []
    for uid in all_ids:
        records.append({
            "user_id":         uid,
            "in_degree":       in_deg.get(uid, 0),
            "out_degree":      out_deg.get(uid, 0),
            "degree_ratio":    in_deg.get(uid, 0) / max(out_deg.get(uid, 1), 1),
            "pagerank":        pagerank.get(uid, 0.0),
            "hits_authority":  hits_auth.get(uid, 0.0),
            "hits_hub":        hits_hub.get(uid, 0.0),
            "betweenness":     between.get(uid, 0.0),
            "clustering_coef": clust.get(uid, 0.0),
        })
    return pd.DataFrame(records)


def compute_tweet_features(tweets, user_ids):
    print("  Aggregating tweet engagement metrics per user...")
    if len(tweets) == 0:
        return pd.DataFrame({"user_id": list(user_ids)})
    grp = tweets.groupby("user_id").agg(
        tweet_count_actual=("tweet_id",      "count"),
        avg_retweets=      ("retweet_count", "mean"),
        avg_likes=         ("like_count",    "mean"),
        avg_replies=       ("reply_count",   "mean"),
        avg_hashtags=      ("hashtag_count", "mean"),
        avg_mentions=      ("mention_count", "mean"),
    ).reset_index()
    grp["total_engagement"] = grp["avg_retweets"] + grp["avg_likes"] + grp["avg_replies"]
    base = pd.DataFrame({"user_id": list(user_ids)})
    return base.merge(grp, on="user_id", how="left").fillna(0)


def compute_profile_features(users):
    print("  Deriving profile-level features...")
    df = users.copy()
    df["follower_following_ratio"] = df["followers_count"] / df["following_count"].clip(lower=1)
    df["tweets_per_day"]      = df["tweet_count"]  / df["account_age_days"].clip(lower=1)
    df["listed_per_follower"] = df["listed_count"] / df["followers_count"].clip(lower=1)
    keep = ["user_id","verified","followers_count","following_count",
            "tweet_count","listed_count","account_age_days",
            "follower_following_ratio","tweets_per_day","listed_per_follower"]
    return df[keep]


def build_influence_label(features):
    """Composite score: top 20% -> influential(1), rest -> 0."""
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    score = (
        0.30 * norm(features["pagerank"]) +
        0.25 * norm(features["followers_count"]) +
        0.25 * norm(features["total_engagement"]) +
        0.20 * norm(features["betweenness"])
    )
    features = features.copy()
    features["influence_score"] = score
    features["influential"]     = (score >= score.quantile(0.80)).astype(int)
    dist = features["influential"].value_counts().to_dict()
    print(f"  Labels: 1=influential={dist.get(1,0)}  |  0=non-influential={dist.get(0,0)}")
    return features


def engineer_features():
    print("=" * 55)
    print(" FEATURE ENGINEERING")
    print("=" * 55)
    print("[1/5] Loading raw data...")
    users, edges, tweets = load_data()
    print("[2/5] Building directed social graph...")
    G = build_graph(edges)
    print("[3/5] Network centrality metrics...")
    net_feats = compute_network_features(G, users["user_id"])
    print("[4/5] Tweet engagement + profile features...")
    tweet_feats   = compute_tweet_features(tweets, users["user_id"])
    profile_feats = compute_profile_features(users)
    print("[5/5] Merging and labelling...")
    features = (
        profile_feats
        .merge(net_feats,   on="user_id", how="left")
        .merge(tweet_feats, on="user_id", how="left")
    )
    features = build_influence_label(features)
    out_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    print(f"\nFeature matrix: {features.shape[0]} users x {features.shape[1]} columns")
    print(f"Saved -> {out_path}")
    return features, G


if __name__ == "__main__":
    engineer_features()