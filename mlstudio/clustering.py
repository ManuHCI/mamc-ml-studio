"""Finding groups of similar patients (unsupervised learning)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_check import is_num

CLUSTER_METHODS = {
    "kmeans": dict(
        name="K-means",
        one_line="Places k 'centres' and gives every patient to the nearest centre; the centres move until the groups settle.",
        analogy="Like setting up k OPD counters and letting each patient walk to the closest one, then moving each counter "
                "to the middle of its queue - again and again.",
        how=["Choose the number of groups k.", "Put k centres at random patients.",
             "Assign every patient to the nearest centre.", "Move each centre to the average of its patients.",
             "Repeat the last two steps until nothing changes."],
        good="Fast; works well for round, similarly sized groups.",
        bad="You must choose k; struggles with odd-shaped groups; sensitive to scaling and outliers."),
    "hier": dict(
        name="Hierarchical (agglomerative)",
        one_line="Starts with every patient alone and keeps merging the two most similar groups, building a family tree.",
        analogy="Like building a family tree (dendrogram) of patients: siblings first, then cousins, then whole families.",
        how=["Every patient starts as a group of one.", "The two closest groups are merged.",
             "This repeats until everyone is in one group.", "You 'cut' the tree at the level that gives k groups."],
        good="Shows how groups relate to each other; no random start.",
        bad="Slow for very large datasets; merges cannot be undone."),
    "dbscan": dict(
        name="DBSCAN (density)",
        one_line="Finds crowded regions of similar patients; patients in empty areas are called 'noise'.",
        analogy="Like spotting clusters of cases on an outbreak map: dense pockets become groups, isolated cases stay alone.",
        how=["For each patient, count neighbours within a distance 'eps'.",
             "Patients with enough neighbours (min_samples) form the core of a group.",
             "Groups grow by joining neighbouring core patients.", "Patients that belong to no dense region are labelled noise."],
        good="No need to choose k; finds odd-shaped groups; flags unusual patients.",
        bad="Sensitive to eps; struggles when groups have different densities."),
}


def prepare(df: pd.DataFrame, features: list[str]):
    X = df[features].copy()
    num = [c for c in features if is_num(X[c])]
    cat = [c for c in features if c not in num]
    parts = []
    if num:
        Xn = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X[num].astype(float)), columns=num, index=X.index)
        parts.append(Xn)
    if cat:
        Xc = pd.get_dummies(X[cat].astype(str).replace("nan", np.nan), dummy_na=False).astype(float)
        parts.append(Xc)
    Xall = pd.concat(parts, axis=1)
    Z = StandardScaler().fit_transform(Xall.values)
    return Xall, Z


def scan_k(Z, kmax=8, seed=42):
    ks, inertias, sils = [], [], []
    for k in range(2, min(kmax, len(Z) - 1) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        ks.append(k)
        inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(Z, km.labels_)) if len(set(km.labels_)) > 1 else np.nan)
    return ks, inertias, sils


def run(method: str, Z, k=3, linkage="ward", eps=1.5, min_samples=5, seed=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "kmeans":
            lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        elif method == "hier":
            lab = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(Z)
        else:
            lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    # order groups by size so 'Group 1' is the largest
    s = pd.Series(lab)
    order = [g for g in s[s >= 0].value_counts().index]
    names = {g: f"Group {i + 1}" for i, g in enumerate(order)}
    names[-1] = "Noise"
    labels = np.array([names[v] for v in lab], dtype=object)
    sil = np.nan
    valid = lab >= 0
    if len(set(lab[valid])) > 1:
        sil = float(silhouette_score(Z[valid], lab[valid]))
    return labels, sil


def pca_coords(Z):
    p = PCA(n_components=2, random_state=0)
    c = p.fit_transform(Z)
    return c, p.explained_variance_ratio_


def profile(df: pd.DataFrame, Xall: pd.DataFrame, labels) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean of each variable per group (original units) and standardised difference from the overall mean."""
    d = Xall.copy()
    d["Group"] = labels
    means = d.groupby("Group").mean()
    sizes = d["Group"].value_counts()
    sd = Xall.std(ddof=0).replace(0, 1)
    z = (means - Xall.mean()) / sd
    means.insert(0, "Patients", sizes)
    return means, z


def describe_groups(z: pd.DataFrame, top=3) -> dict:
    out = {}
    for g, row in z.iterrows():
        if g == "Noise":
            out[g] = "Patients that do not fit any dense group (possible outliers)."
            continue
        hi = row[row > 0.5].sort_values(ascending=False).head(top)
        lo = row[row < -0.5].sort_values().head(top)
        parts = []
        if len(hi):
            parts.append("higher " + ", ".join(hi.index))
        if len(lo):
            parts.append("lower " + ", ".join(lo.index))
        out[g] = ("Compared with the average patient: " + "; ".join(parts) + ".") if parts else "Close to the average patient on most variables."
    return out


def dendrogram_figure(Z, max_leaves=40):
    """Matplotlib dendrogram (truncated) for hierarchical clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    L = linkage(Z, method="ward")
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
    dendrogram(L, truncate_mode="lastp", p=max_leaves, ax=ax, color_threshold=None, above_threshold_color="#52514e",
               leaf_font_size=8)
    ax.set_title("Dendrogram (family tree of patients)", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance when merged")
    ax.set_xlabel("Patients / small groups (numbers in brackets = patients)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
