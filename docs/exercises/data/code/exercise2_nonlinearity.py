"""Exercise 2 — Non-Linearity in Higher Dimensions.

Gera o Dataset I (gaussianas deslocadas 5D, item A) e o Dataset II (cascas
concêntricas 5D, item B), projeta ambos em 2D via PCA, calcula distância entre
centros e raio por classe, e salva as figuras 4-5 em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise2_nonlinearity.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

N_SAMPLES = 500

MU_A = np.zeros(5)
COV_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

MU_B = np.full(5, 1.5)
COV_B = np.array([
    [ 1.5, -0.7,  0.2, 0.0, 0.0],
    [-0.7,  1.5,  0.4, 0.0, 0.0],
    [ 0.2,  0.4,  1.5, 0.6, 0.0],
    [ 0.0,  0.0,  0.6, 1.5, 0.3],
    [ 0.0,  0.0,  0.0, 0.3, 1.5],
])

RHO_CORE_MEAN, RHO_CORE_STD = 2.0, 0.4
RHO_SHELL_MEAN, RHO_SHELL_STD = 5.0, 0.4


def sample_unit_sphere(n_samples, dim, rng):
    """Direção uniforme na esfera unitária de R^dim: normaliza um vetor gaussiano."""
    v = rng.normal(size=(n_samples, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def plot_figure4(X_ds1_2d, y_ds1, ev1, X_ds2_2d, y_ds2, ev2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for cls, label in zip([0, 1], ["Classe A", "Classe B"]):
        mask = y_ds1 == cls
        axes[0].scatter(X_ds1_2d[mask, 0], X_ds1_2d[mask, 1], color=colors[cls],
                        alpha=0.6, label=label)
    axes[0].set_title(f"Dataset I — Gaussianas deslocadas (PCA, EV={ev1.sum():.2f})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    for cls, label in zip([0, 1], ["Classe C (núcleo)", "Classe D (casca)"]):
        mask = y_ds2 == cls
        axes[1].scatter(X_ds2_2d[mask, 0], X_ds2_2d[mask, 1], color=colors[cls + 2],
                        alpha=0.6, label=label)
    axes[1].set_title(f"Dataset II — Cascas concêntricas (PCA, EV={ev2.sum():.2f})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    fig.suptitle("Figura 4 — Projeção PCA para 2D")
    return fig


def plot_figure5(radius_A, radius_B, radius_C, radius_D):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(radius_A, bins=25, alpha=0.6, label="Classe A", color="tab:blue")
    axes[0].hist(radius_B, bins=25, alpha=0.6, label="Classe B", color="tab:orange")
    axes[0].set_title("Dataset I — Histograma do raio ||x||")
    axes[0].set_xlabel("||x||")
    axes[0].set_ylabel("Frequência")
    axes[0].legend()

    axes[1].hist(radius_C, bins=25, alpha=0.6, label="Classe C (núcleo)", color="tab:green")
    axes[1].hist(radius_D, bins=25, alpha=0.6, label="Classe D (casca)", color="tab:red")
    axes[1].set_title("Dataset II — Histograma do raio ||x||")
    axes[1].set_xlabel("||x||")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    fig.suptitle("Figura 5 — Distribuição do raio por classe")
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Item A: Dataset I ---
    X_A = RNG.multivariate_normal(MU_A, COV_A, size=N_SAMPLES)
    X_B = RNG.multivariate_normal(MU_B, COV_B, size=N_SAMPLES)
    X_ds1 = np.vstack([X_A, X_B])
    y_ds1 = np.concatenate([np.zeros(N_SAMPLES), np.ones(N_SAMPLES)])

    # --- Item B: Dataset II ---
    u_C = sample_unit_sphere(N_SAMPLES, 5, RNG)
    rho_C = RNG.normal(RHO_CORE_MEAN, RHO_CORE_STD, size=N_SAMPLES)
    X_C = rho_C[:, None] * u_C  # (1)!

    u_D = sample_unit_sphere(N_SAMPLES, 5, RNG)
    rho_D = RNG.normal(RHO_SHELL_MEAN, RHO_SHELL_STD, size=N_SAMPLES)
    X_D = rho_D[:, None] * u_D

    X_ds2 = np.vstack([X_C, X_D])
    y_ds2 = np.concatenate([np.zeros(N_SAMPLES), np.ones(N_SAMPLES)])

    # --- Item C: PCA, distâncias, raios ---
    pca1 = PCA(n_components=2)
    X_ds1_2d = pca1.fit_transform(X_ds1)
    ev1 = pca1.explained_variance_ratio_

    pca2 = PCA(n_components=2)
    X_ds2_2d = pca2.fit_transform(X_ds2)
    ev2 = pca2.explained_variance_ratio_

    print(f"Dataset I  — variância explicada PC1+PC2: {ev1.sum():.3f}")
    print(f"Dataset II — variância explicada PC1+PC2: {ev2.sum():.3f}")

    fig4 = plot_figure4(X_ds1_2d, y_ds1, ev1, X_ds2_2d, y_ds2, ev2)
    fig4.savefig(FIGURES / "fig04-pca-projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)  # (2)!

    dist_ds1 = np.linalg.norm(X_A.mean(axis=0) - X_B.mean(axis=0))
    dist_ds2 = np.linalg.norm(X_C.mean(axis=0) - X_D.mean(axis=0))
    print(f"Distância entre centros — Dataset I:  {dist_ds1:.3f}")
    print(f"Distância entre centros — Dataset II: {dist_ds2:.3f}")

    radius_A, radius_B = np.linalg.norm(X_A, axis=1), np.linalg.norm(X_B, axis=1)
    radius_C, radius_D = np.linalg.norm(X_C, axis=1), np.linalg.norm(X_D, axis=1)

    fig5 = plot_figure5(radius_A, radius_B, radius_C, radius_D)
    fig5.savefig(FIGURES / "fig05-radius-histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)


if __name__ == "__main__":
    main()
