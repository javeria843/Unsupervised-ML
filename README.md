# Unsupervised Clustering — K-Means, Hierarchical, DBSCAN

A compact project README for exploring three unsupervised clustering algorithms (K-Means, Hierarchical/Agglomerative, DBSCAN) with data preprocessing, parameter selection tips, evaluation, and visualization examples.

---

## ✅ Project Overview

This repo/demo demonstrates how to apply **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN** to find structure in unlabeled data. It includes preprocessing, parameter selection (Elbow, dendrogram, k-distance), evaluation (Silhouette, Davies–Bouldin), and plotting examples.

---

## 📂 File Structure (suggested)

```
.
├─ data/
│  └─ sample.csv
├─ notebooks/
│  └─ clustering_demo.ipynb
├─ src/
│  ├─ preprocess.py
│  ├─ run_kmeans.py
│  ├─ run_hierarchical.py
│  └─ run_dbscan.py
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 🛠️ Requirements

* Python 3.8+
* Major packages (also in `requirements.txt`):

```
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
joblib
```

---

## 🚀 Quick Setup

1. Create and activate virtualenv (optional)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Open `notebooks/clustering_demo.ipynb` or run scripts in `src/`.

---

## 🧹 Data preprocessing (recommended)

1. Load data with `pandas`.
2. Handle missing values (`dropna()` or impute).
3. Encode categorical features (one-hot or ordinal).
4. Scale numeric features — **StandardScaler** (zero mean, unit variance) or **MinMaxScaler** depending on algorithm:

   * K-Means, DBSCAN: scaling important (use StandardScaler or MinMaxScaler)
   * Hierarchical: scale to keep feature contributions balanced
5. Optionally reduce dimensionality (PCA / t-SNE / UMAP) for visualization.

Example (in `preprocess.py`):

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(df, numeric_cols):
    imputer = SimpleImputer(strategy='median')
    X_num = imputer.fit_transform(df[numeric_cols])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    return X_scaled
```

---

## 📐 Parameter selection & tips

### K-Means

* Choose `k` with the **Elbow Method** (plot inertia vs k) or **Silhouette score** (higher is better).
* Run multiple `n_init` restarts (e.g., `n_init=10`) to avoid bad local minima.

### Hierarchical (Agglomerative)

* Plot a **dendrogram** (scipy) to see merge distances and choose cut level.
* Linkage choices: `ward` (variance-minimizing) for Euclidean; `complete`, `average` for other shapes.

### DBSCAN

* Tune `eps` via the **k-distance plot** (plot distance to k-th nearest neighbor, often k = `min_samples`).
* `min_samples` rule-of-thumb: `min_samples = 2 * n_features` (or 4–10).
* DBSCAN identifies noise points (label = -1).

---

## 🔎 Evaluation metrics

* **Silhouette score** — measures cohesion vs separation (range -1..1). Good for comparing clusterings.
* **Davies–Bouldin Index** — lower is better.
* **Cluster sizes** — check for empty or extremely small clusters.
* For DBSCAN check number/percentage of noise points.

Example:

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

sil = silhouette_score(X, labels)
db = davies_bouldin_score(X, labels)
```

---

## 📈 Example usage snippets

### K-Means

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

### Hierarchical (Agglomerative)

```python
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X_scaled)
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)
```

---

## 📊 Visualization ideas

* 2D scatter (PCA / TSNE reduced) colored by `labels`.
* Pairplot or cluster centroids (for low-dim data).
* Dendrogram for hierarchical clustering.
* Elbow plot (K vs inertia), silhouette plot, and k-distance plot for DBSCAN.

Quick scatter example:

```python
import matplotlib.pyplot as plt
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='tab10', s=30)
plt.title("Clusters")
plt.show()
```

---

## ✅ Good practices / checks

* Always scale features before K-Means and DBSCAN.
* Try multiple `k` and initialization seeds for K-Means.
* Validate clusters by domain knowledge, not only metrics.
* Watch for high percentage of noise in DBSCAN — adjust `eps` or transform features.

---

## ✍️ Contributing

* Open issues for bugs/feature requests.
* Add new example datasets, parameter-search utilities, or visualization modules.

---

## 📞 Contact / Author

* Created by: *Your Name / Your GitHub*
* Email: [your.email@example.com](mailto:your.email@example.com)

---

## 📜 License

Include your preferred license (e.g., MIT).

