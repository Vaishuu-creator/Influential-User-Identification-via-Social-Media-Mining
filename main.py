"""
main.py  -  Influential User Identification via Social Media Mining
====================================================================
Pipeline orchestrator - runs the full project end-to-end:

  Step 1: data_generator.py     - generate / simulate raw social network data
  Step 2: feature_engineering.py - compute graph + engagement features
  Step 3: model_training.py      - train 4 classifiers, save best model
  Step 4: influence_analysis.py  - rank users, detect communities, visualise

Dataset Reference:
  Kaggle: "Twitter User Data" (TwitterFriends)
  URL:    https://www.kaggle.com/datasets/hwassner/TwitterFriends
"""

import time

def run():
    print("\n" + "=" * 60)
    print("  INFLUENTIAL USER IDENTIFICATION VIA SOCIAL MEDIA MINING")
    print("=" * 60)
    print("  Dataset: Twitter User Data (Kaggle - hwassner/TwitterFriends)")
    print("  URL    : https://www.kaggle.com/datasets/hwassner/TwitterFriends")
    print("=" * 60 + "\n")

    total_start = time.time()

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    print("\n>>> STEP 1: DATA GENERATION")
    t = time.time()
    from data_generator import (N_USERS, N_EDGES, N_TWEETS)
    import data_generator as dg
    # Re-run by importing the module (it auto-runs on import for this project)
    import importlib, sys
    # Execute the generator
    import subprocess, os
    import pandas as pd, numpy as np, random
    random.seed(42); np.random.seed(42)
    exec(open(os.path.join(os.path.dirname(__file__), "data_generator.py")).read())
    print(f"    Elapsed: {time.time()-t:.1f}s")

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    print("\n>>> STEP 2: FEATURE ENGINEERING")
    t = time.time()
    from feature_engineering import engineer_features
    features, G = engineer_features()
    print(f"    Elapsed: {time.time()-t:.1f}s")

    # ── Step 3: Model Training ────────────────────────────────────────────────
    print("\n>>> STEP 3: MODEL TRAINING & EVALUATION")
    t = time.time()
    from model_training import train_and_evaluate
    results, models, X, y = train_and_evaluate()
    print(f"    Elapsed: {time.time()-t:.1f}s")

    # ── Step 4: Influence Analysis ────────────────────────────────────────────
    print("\n>>> STEP 4: INFLUENCE ANALYSIS & VISUALISATION")
    t = time.time()
    from influence_analysis import run_analysis
    ranked = run_analysis()
    print(f"    Elapsed: {time.time()-t:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total runtime : {total:.1f}s")
    print(f"  Users analysed: {len(ranked)}")
    inf_count = ranked['influential'].sum()
    print(f"  Influential   : {inf_count} ({100*inf_count/len(ranked):.0f}%)")
    print(f"  Best model AUC: {max(r['roc_auc'] for r in results.values()):.3f}")
    print("\n  Output files:")
    print("    data/features.csv         - full feature matrix")
    print("    data/ranked_users.csv     - users ranked by influence")
    print("    models/best_model.pkl     - serialised best classifier")
    print("    models/results.csv        - model comparison metrics")
    print("    outputs/roc_curves.png    - ROC comparison plot")
    print("    outputs/confusion_matrix.png")
    print("    outputs/feature_importance.png")
    print("    outputs/top_influencers.png")
    print("    outputs/network_communities.png")
    print("    outputs/topic_distribution.png")
    print("=" * 60)


if __name__ == "__main__":
    run()