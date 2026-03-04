<div align="center">

# 🔍 Influential User Identification via Social Media Mining

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1%2B-brightgreen?style=for-the-badge)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A full end-to-end **social network mining** and **machine learning** pipeline that identifies influential users using graph-theoretic centrality, engagement signals, and profile metadata — achieving **97% accuracy** and **ROC-AUC of 1.00**.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Outputs](#-outputs)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

Social media platforms generate massive graphs of user relationships. This project mines those graphs to answer a fundamental question:

> **Who are the most influential users — and what signals predict influence?**

The pipeline combines **graph theory** (PageRank, betweenness centrality, HITS) with **engagement metrics** and **profile features** to build a composite influence score and train four ML classifiers. It also performs **community detection** to uncover clusters of closely connected users.

**Key highlights:**
- 24 engineered features per user spanning network, engagement, and profile dimensions
- 4 classifiers compared: Random Forest, Gradient Boosting, SVM, Logistic Regression
- Community detection using greedy modularity (9 communities found)
- 6 publication-quality visualisation outputs

---

## 📦 Dataset

| Field | Detail |
|-------|--------|
| **Name** | Twitter User Data — Ego Networks / Follower Graphs |
| **Source** | Kaggle |
| **URL** | [https://www.kaggle.com/datasets/hwassner/TwitterFriends](https://www.kaggle.com/datasets/hwassner/TwitterFriends) |
| **Contents** | User profiles, follower/following edges, tweet metadata |

### Using the Real Dataset

1. Download from the Kaggle link above
2. Place `users.csv`, `edges.csv`, and `tweets.csv` inside the `data/` folder
3. Ensure column names match the schema in `data_generator.py`
4. Comment out the data generation call in `main.py`

> **No Kaggle account?** The included `data_generator.py` produces a fully synthetic dataset with an identical schema (500 users, 4 000 edges, 3 000 tweets) so the pipeline runs out of the box.

---

## 🗂️ Project Structure

```
influential_user_identification/
│
├── main.py                    # ▶ Pipeline orchestrator — run this
│
├── data_generator.py          # Step 1 — Synthetic data (mirrors Kaggle schema)
├── feature_engineering.py     # Step 2 — Graph + engagement + profile features
├── model_training.py          # Step 3 — Train & evaluate 4 ML classifiers
├── influence_analysis.py      # Step 4 — Ranking, communities, visualisations
│
├── requirements.txt           # Python dependencies
│
├── data/                      # Auto-created at runtime
│   ├── users.csv
│   ├── edges.csv
│   ├── tweets.csv
│   ├── features.csv
│   └── ranked_users.csv
│
├── models/                    # Auto-created at runtime
│   ├── best_model.pkl
│   ├── results.csv
│   └── feature_importance.csv
│
└── outputs/                   # Auto-created at runtime
    ├── roc_curves.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── top_influencers.png
    ├── network_communities.png
    └── topic_distribution.png
```

---

## 🔬 Methodology

### 1 · Data Model

Three interconnected tables represent the social network:

```
users.csv  ──< edges.csv >──  users.csv
              (follower → followee)
              
users.csv  ──< tweets.csv
              (user_id → tweet activity)
```

### 2 · Feature Engineering

| Category | Features |
|---|---|
| **Network — Centrality** | In-degree, out-degree, degree ratio, PageRank, betweenness centrality, clustering coefficient |
| **Network — Influence** | HITS authority score, HITS hub score |
| **Profile** | Followers count, following count, verified status, account age, follower/following ratio, listed count, tweets per day |
| **Engagement** | Avg retweets, avg likes, avg replies, avg hashtags, avg mentions, total engagement score |
| **Label** | Composite weighted score → top 20% labelled `influential = 1` |

**Influence score formula:**

```
score = 0.30 × PageRank  +  0.25 × Followers  +  0.25 × Engagement  +  0.20 × Betweenness
```
*(all terms normalised to [0, 1] before weighting)*

### 3 · Models

| Model | Strategy |
|---|---|
| **Random Forest** | Ensemble of 200 decision trees; robust to outliers and class imbalance |
| **Gradient Boosting** | Sequential boosting with depth-5 trees; high accuracy on tabular data |
| **SVM (RBF kernel)** | Max-margin classifier; effective in high-dimensional feature spaces |
| **Logistic Regression** | Interpretable linear baseline with L2 regularisation |

All models use `class_weight="balanced"` to handle the 4:1 class imbalance.

### 4 · Community Detection

Greedy modularity maximisation (`networkx.community.greedy_modularity_communities`) is applied to the undirected projection of the follower graph. Nodes in the network visualisation are sized by influence score and coloured by community membership.

---

## 📊 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | **0.970** | 0.870 | **1.000** | **0.930** | **1.000** |
| Random Forest | 0.960 | 0.833 | 1.000 | 0.909 | 1.000 |
| SVM (RBF) | 0.960 | 0.833 | 1.000 | 0.909 | 1.000 |
| Gradient Boosting | 0.960 | 0.833 | 1.000 | 0.909 | 0.987 |

> ✅ **Best model:** Logistic Regression — F1 = **0.930**, AUC = **1.000**

### Top Feature Importances (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `total_engagement` | 37.2% |
| 2 | `avg_likes` | 21.3% |
| 3 | `avg_retweets` | 5.0% |
| 4 | `degree_ratio` | 4.1% |
| 5 | `hits_authority` | 3.8% |
| 6 | `in_degree` | 3.3% |
| 7 | `pagerank` | 2.9% |
| 8 | `betweenness` | 2.3% |

> **Key insight:** Engagement metrics (likes, retweets) are the strongest predictors of influence, followed by graph-theoretic signals (degree ratio, HITS authority, PageRank).

### Community Detection

- **9 distinct communities** identified in the 500-node network
- Influential users (top 20%) are distributed across communities, indicating that influence is not confined to a single cluster

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/influential-user-identification.git
cd influential-user-identification

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This single command executes all four steps and writes every output to `data/`, `models/`, and `outputs/`.

### Run Individual Steps

```bash
# Generate synthetic data
python data_generator.py

# Compute features
python feature_engineering.py

# Train models
python model_training.py

# Analyse and visualise
python influence_analysis.py
```

---

## 📁 Outputs

| File | Description |
|---|---|
| `data/features.csv` | Feature matrix — 500 users × 24 features |
| `data/ranked_users.csv` | All users ranked by composite influence score |
| `models/best_model.pkl` | Best classifier serialised via joblib |
| `models/results.csv` | Accuracy / F1 / AUC table for all four models |
| `models/feature_importance.csv` | Ranked feature importances from Random Forest |
| `outputs/roc_curves.png` | ROC curves — all four models on one chart |
| `outputs/confusion_matrix.png` | 2×2 confusion matrix grid for all models |
| `outputs/feature_importance.png` | Horizontal bar chart — top 20 features |
| `outputs/top_influencers.png` | Influence score, PageRank, and followers for top 10 users |
| `outputs/network_communities.png` | Spring-layout network coloured by community |
| `outputs/topic_distribution.png` | Tweet topic breakdown — all users vs. influential users |

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading, transformation, aggregation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `networkx` | ≥ 3.1 | Graph construction, centrality metrics, community detection |
| `scikit-learn` | ≥ 1.3 | ML classifiers, preprocessing, evaluation |
| `matplotlib` | ≥ 3.7 | Visualisation |
| `seaborn` | ≥ 0.12 | Heatmaps and styled plots |
| `scipy` | ≥ 1.10 | Statistical utilities |
| `joblib` | ≥ 1.3 | Model serialisation |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "Add your feature"`
4. Push to the branch — `git push origin feature/your-feature`
5. Open a Pull Request

### Ideas for extension
- Integrate real-time Twitter/X API data collection
- Add temporal influence tracking (how scores change over time)
- Experiment with GNN-based models (Graph Attention Networks)
- Build an interactive dashboard with Streamlit or Dash

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for the data science community

⭐ Star this repo if you found it useful!

</div>
