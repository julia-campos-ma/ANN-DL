"""
Stage 4B -- three 2D projections of the preprocessed TRAINING features, coloured by the
target: PCA (variance and loadings), t-SNE (two perplexities) and UMAP (two
n_neighbors). t-SNE and UMAP run on a 5,000-row sample drawn with the fixed seed; PCA
is fitted on the whole training set and drawn on the same sample. Figures 13 to 15.

Run from the repository root (about 1 minute):
    python docs/projects/eda/code/stage4_projections.py
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from data import FIGURES, SEED, TARGET, load_split
from pipeline import fit_transform_splits

warnings.filterwarnings("ignore", category=UserWarning)  # umap: spectral init fallback, n_jobs

COLORS = {0: "#4C72B0", 1: "#DD8452"}
LABELS = {0: "não cancelada", 1: "cancelada"}
SAMPLE = 5000
PERPLEXITIES = (5, 50)
NEIGHBORS = (5, 50)


def neighbourhood_purity(emb, y, k=10):
    """Share of the k nearest neighbours (in the 2D embedding) that have the same label.

    A descriptive statistic of how separated the classes look in the picture; a random
    arrangement gives p^2 + (1-p)^2 for a class share p.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]
    return float((y[idx] == y[:, None]).mean())


def describe_region(name, mask, y, sub):
    """What is inside a region of a 2D picture: cancel rate and the dominant levels."""
    rows = sub.loc[mask]
    dup = rows[[c for c in rows.columns if c != "arrival_date"]].duplicated(keep=False).mean()
    print(f"  {name}: {int(mask.sum())} points, cancel rate {y[mask].mean():.1%}, "
          f"median lead_time {rows.lead_time.median():.0f} d, exact duplicates {dup:.0%}; "
          + "; ".join(f"{col} " + ", ".join(f"{k} {v:.0%}" for k, v in
                                           rows[col].value_counts(normalize=True).head(2).items())
                      for col in ("deposit_type", "market_segment", "hotel")))


def scatter(ax, emb, y, title):
    for c in (0, 1):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, color=COLORS[c], label=LABELS[c], rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    method = title.split(",")[0]
    ax.set_xlabel(f"{method} 1", fontsize=8)
    ax.set_ylabel(f"{method} 2", fontsize=8)


# --8<-- [start:pca]
def run_pca(X_train, names, y_sample, X_sample):
    pca = PCA(random_state=SEED).fit(X_train)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA on {X_train.shape[1]} preprocessed features:")
    print(f"  PC1 {pca.explained_variance_ratio_[0]:.2%}, PC2 {pca.explained_variance_ratio_[1]:.2%}, "
          f"PC1+PC2 {cum[1]:.2%}; components for 50% / 80% / 90%: "
          f"{int(np.searchsorted(cum, 0.5)) + 1} / {int(np.searchsorted(cum, 0.8)) + 1} / "
          f"{int(np.searchsorted(cum, 0.9)) + 1}")
    for k in (0, 1):
        load = pd.Series(pca.components_[k], index=names)
        top = load.reindex(load.abs().sort_values(ascending=False).index).head(6)
        print(f"  PC{k + 1} loadings: " + ", ".join(f"{n} {v:+.2f}" for n, v in top.items()))
    emb = pca.transform(X_sample)[:, :2]
    print(f"  neighbourhood purity (k=10) in the PC1-PC2 plane: {neighbourhood_purity(emb, y_sample):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].plot(np.arange(1, 41), cum[:40], marker="o", ms=3, color="#4C72B0")
    axes[0].axhline(0.9, color="gray", ls=":")
    axes[0].set_xlabel("número de componentes")
    axes[0].set_ylabel("variância explicada acumulada")
    axes[0].set_title(f"PC1 + PC2 = {cum[1]:.1%}; 90% com {int(np.searchsorted(cum, 0.9)) + 1} componentes", fontsize=9)
    scatter(axes[1], emb, y_sample, f"PC1 × PC2 (amostra de {len(y_sample)} reservas de treino)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8, markerscale=3)
    fig.suptitle("Figura 13 — PCA das features pré-processadas (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13-pca.png", dpi=150)
    plt.close(fig)
    return emb
# --8<-- [end:pca]


# --8<-- [start:nonlinear]
def run_tsne(X_sample, y_sample, sub):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, perp in zip(axes, PERPLEXITIES):
        t0 = time.time()
        emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=SEED).fit_transform(X_sample)
        purity = neighbourhood_purity(emb, y_sample)
        print(f"t-SNE perplexity={perp}: purity {purity:.3f}  ({time.time() - t0:.0f} s)")
        scatter(ax, emb, y_sample, f"t-SNE, perplexidade = {perp} · pureza de vizinhança {purity:.2f}")
        if perp == 50:
            describe_region("top 10% of the picture (y)", emb[:, 1] > np.percentile(emb[:, 1], 90), y_sample, sub)
            describe_region("right 8% of the picture (x)", emb[:, 0] > np.percentile(emb[:, 0], 92), y_sample, sub)
    axes[0].legend(fontsize=8, markerscale=3)
    fig.suptitle(f"Figura 14 — t-SNE da amostra de {len(y_sample)} reservas de treino, duas perplexidades")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14-tsne.png", dpi=150)
    plt.close(fig)


def run_umap(X_sample, y_sample, sub):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, nn in zip(axes, NEIGHBORS):
        t0 = time.time()
        emb = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=SEED).fit_transform(X_sample)
        purity = neighbourhood_purity(emb, y_sample)
        print(f"UMAP n_neighbors={nn}: purity {purity:.3f}  ({time.time() - t0:.0f} s)")
        scatter(ax, emb, y_sample, f"UMAP, n_neighbors = {nn} · pureza de vizinhança {purity:.2f}")
        if nn == 50:
            # Islands = connected groups of points (DBSCAN on the 2D picture, no labels used).
            labels = DBSCAN(eps=0.6, min_samples=20).fit_predict(emb)
            groups = pd.Series(labels[labels >= 0]).value_counts()
            print(f"  DBSCAN finds {len(groups)} islands of >= 20 points; the largest has {groups.iloc[0]} points")
            for lab, size in groups.items():
                m = labels == lab
                if size < groups.iloc[0] and y_sample[m].mean() >= 0.9:
                    describe_region(f"island of {size} points with cancel rate >= 90%", m, y_sample, sub)
    axes[0].legend(fontsize=8, markerscale=3)
    fig.suptitle(f"Figura 15 — UMAP da amostra de {len(y_sample)} reservas de treino, dois n_neighbors")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig15-umap.png", dpi=150)
    plt.close(fig)
# --8<-- [end:nonlinear]


if __name__ == "__main__":
    train, test, cut = load_split()
    X_train, X_test, names, _ = fit_transform_splits(train, test)
    y_train = train[TARGET].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), size=SAMPLE, replace=False)
    X_sample, y_sample = X_train[idx], y_train[idx]
    sub = train.iloc[idx].reset_index(drop=True)
    p = y_sample.mean()
    print(f"sample: {SAMPLE} training rows, cancel rate {p:.2%}; "
          f"purity of a random arrangement = {p**2 + (1 - p)**2:.3f}; "
          f"purity in the original {X_sample.shape[1]}-D preprocessed space = "
          f"{neighbourhood_purity(X_sample, y_sample):.3f}; "
          f"sample points with an identical copy in the sample: "
          f"{sub[[c for c in sub.columns if c != 'arrival_date']].duplicated(keep=False).mean():.1%}")
    run_pca(X_train, names, y_sample, X_sample)
    run_tsne(X_sample, y_sample, sub)
    run_umap(X_sample, y_sample, sub)
    print("\nFigures written to", FIGURES)
