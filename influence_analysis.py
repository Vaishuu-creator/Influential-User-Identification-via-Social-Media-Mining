"""
influence_analysis.py  -  Influential User Identification via Social Media Mining
==================================================================================
Post-model analysis:
  - Ranks users by composite influence score
  - Profiles the top-10 most influential users
  - Community detection using Louvain-style greedy modularity
  - Topic analysis per community
  - Saves ranked users and visualisations

Outputs:
  data/ranked_users.csv
  outputs/top_influencers.png
  outputs/network_communities.png
  outputs/topic_distribution.png
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all():
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    users    = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    edges    = pd.read_csv(os.path.join(DATA_DIR, "edges.csv"))
    tweets   = pd.read_csv(os.path.join(DATA_DIR, "tweets.csv"))
    return features, users, edges, tweets


def rank_users(features, users):
    ranked = features.merge(users[["user_id","screen_name","verified","favourite_topic"]],
                             on="user_id", how="left")
    ranked = ranked.sort_values("influence_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked.to_csv(os.path.join(DATA_DIR, "ranked_users.csv"), index=False)
    print(f"  Ranked {len(ranked)} users. Top 5:")
    cols = ["rank","screen_name","influence_score","pagerank","followers_count","verified"]
    available = [c for c in cols if c in ranked.columns]
    print(ranked[available].head().to_string(index=False))
    return ranked


def plot_top_influencers(ranked):
    top10 = ranked.head(10).copy()
    top10["label"] = top10["screen_name"].fillna(top10["user_id"].astype(str))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Influence score bar
    bars = axes[0].barh(top10["label"][::-1], top10["influence_score"][::-1],
                         color=plt.cm.RdYlGn(np.linspace(0.3, 1.0, 10)))
    axes[0].set_xlabel("Composite Influence Score")
    axes[0].set_title("Top 10 Influential Users")
    axes[0].grid(axis="x", alpha=0.3)

    # PageRank bar
    axes[1].barh(top10["label"][::-1], top10["pagerank"][::-1],
                  color="#1976D2", alpha=0.8)
    axes[1].set_xlabel("PageRank Score")
    axes[1].set_title("PageRank of Top 10")
    axes[1].grid(axis="x", alpha=0.3)

    # Followers bar
    axes[2].barh(top10["label"][::-1], top10["followers_count"][::-1],
                  color="#388E3C", alpha=0.8)
    axes[2].set_xlabel("Followers Count")
    axes[2].set_title("Followers of Top 10")
    axes[2].grid(axis="x", alpha=0.3)

    plt.suptitle("Influential User Analysis - Top 10 Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "top_influencers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def detect_communities_and_plot(edges, ranked):
    print("  Detecting communities (greedy modularity)...")
    G_und = nx.from_pandas_edgelist(edges, source="follower_id", target="followee_id")
    communities = nx.community.greedy_modularity_communities(G_und)
    comm_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = cid

    ranked["community"] = ranked["user_id"].map(comm_map).fillna(-1).astype(int)
    print(f"  Found {len(communities)} communities")

    # Visualise a subgraph of top-N nodes for clarity
    top_nodes = set(ranked.head(100)["user_id"].tolist())
    H = G_und.subgraph(top_nodes)
    pos = nx.spring_layout(H, seed=42, k=0.5)

    node_colors = [comm_map.get(n, 0) % 20 for n in H.nodes()]
    sizes = []
    score_dict = dict(zip(ranked["user_id"], ranked["influence_score"]))
    for n in H.nodes():
        s = score_dict.get(n, 0)
        sizes.append(100 + 1000 * s)

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(H, pos, node_color=node_colors,
                            cmap=plt.cm.tab20, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(H, pos, alpha=0.15, width=0.5, edge_color="#888888")

    # Label top-5
    top5_ids = set(ranked.head(5)["user_id"].tolist())
    name_dict = dict(zip(ranked["user_id"], ranked["screen_name"]))
    labels = {n: name_dict.get(n, str(n)) for n in H.nodes() if n in top5_ids}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=9, font_weight="bold")

    plt.title("Social Network Communities\n(node size = influence score, colour = community)",
               fontsize=13, fontweight="bold")
    plt.axis("off")
    path = os.path.join(OUTPUT_DIR, "network_communities.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return ranked


def plot_topic_distribution(tweets, ranked):
    merged = tweets.merge(ranked[["user_id","influential"]], on="user_id", how="left")
    merged["influential"] = merged["influential"].fillna(0)

    topic_inf = merged[merged["influential"] == 1]["topic"].value_counts()
    topic_all = merged["topic"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall
    axes[0].bar(topic_all.index, topic_all.values,
                 color=plt.cm.Set3(np.linspace(0, 1, len(topic_all))))
    axes[0].set_title("Tweet Topics - All Users")
    axes[0].set_xlabel("Topic")
    axes[0].set_ylabel("Tweet Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Influential only
    axes[1].bar(topic_inf.index, topic_inf.values,
                 color=plt.cm.Set1(np.linspace(0, 1, len(topic_inf))))
    axes[1].set_title("Tweet Topics - Influential Users Only")
    axes[1].set_xlabel("Topic")
    axes[1].set_ylabel("Tweet Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Topic Distribution Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "topic_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_analysis():
    print("=" * 55)
    print(" INFLUENCE ANALYSIS & VISUALISATION")
    print("=" * 55)
    print("[1/4] Loading data...")
    features, users, edges, tweets = load_all()

    print("[2/4] Ranking users by influence score...")
    ranked = rank_users(features, users)

    print("[3/4] Plotting top influencers...")
    plot_top_influencers(ranked)

    print("[4/4] Community detection and topic analysis...")
    ranked = detect_communities_and_plot(edges, ranked)
    plot_topic_distribution(tweets, ranked)

    print("\nAnalysis complete. All outputs in ./outputs/")
    return ranked


if __name__ == "__main__":
    run_analysis()