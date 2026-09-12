"""Exercise 1 — Point Clouds: Geometry and Spread in 2D.

Gera as 4 nuvens gaussianas do enunciado (item A), as 4 versões em escala
(item B), a tabela de separation ratio, a mixing rate por escala, e salva as
figuras 1-3 em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

PARAMS = {
    0: (2, 3, 0.8, 2.5),
    1: (5, 6, 1.2, 1.9),
    2: (8, 1, 0.9, 0.9),
    3: (15, 4, 0.5, 2.0),
}
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]


def generate_clouds(params, n_per_class, rng):
    """Gera nuvens gaussianas 2D, uma por classe.

    Cada eixo (x, y) é amostrado separadamente porque cada classe tem
    desvio-padrão diferente em x e em y (a nuvem não é isotrópica).
    """
    X_list, y_list = [], []
    for cls, (mx, my, sx, sy) in params.items():
        x = rng.normal(mx, sx, n_per_class)
        yy = rng.normal(my, sy, n_per_class)
        X_list.append(np.column_stack([x, yy]))
        y_list.append(np.full(n_per_class, cls))
    return np.vstack(X_list), np.concatenate(y_list)


def scale_params(params, s):
    """Multiplica os desvios-padrão (não as médias) por s."""
    return {cls: (mx, my, sx * s, sy * s) for cls, (mx, my, sx, sy) in params.items()}


def separation_ratios(params):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), sigma_bar_k = (sigma_kx+sigma_ky)/2."""
    classes = list(params.keys())
    means = {c: np.array(params[c][:2]) for c in classes}
    sigma_bar = {c: (params[c][2] + params[c][3]) / 2 for c in classes}
    results = {}
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = np.linalg.norm(means[ci] - means[cj])
            results[(ci, cj)] = dist / (sigma_bar[ci] + sigma_bar[cj])
    return results


def mixing_rate(X, y, params):
    """Fração de pontos cujo centro de classe mais próximo NÃO é o da própria classe.

    Puramente geométrico — nenhum modelo é treinado.
    """
    classes = sorted(params.keys())
    centers = np.array([params[c][:2] for c in classes])
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.array(classes)[np.argmin(dists, axis=1)]
    return float(np.mean(nearest != y))


def plot_figure1(X, y, params):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for cls, (mx, my, sx, sy) in params.items():
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], color=colors[cls], alpha=0.6, label=f"Classe {cls}")
        ax.scatter(mx, my, color=colors[cls], marker="X", s=200, edgecolor="black", linewidth=1.5)
    ax.set_title("Figura 1 — Nuvens de pontos por classe (s = 1)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    return fig


def plot_figure2(datasets_b, scales):
    all_X = np.vstack([datasets_b[s][0] for s in scales])
    x_min, x_max = all_X[:, 0].min(), all_X[:, 0].max()
    y_min, y_max = all_X[:, 1].min(), all_X[:, 1].max()
    pad_x, pad_y = 0.05 * (x_max - x_min), 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    colors = plt.cm.tab10.colors
    for ax, s in zip(axes, scales):
        X_s, y_s, params_s = datasets_b[s]
        for cls, (mx, my, sx, sy) in params_s.items():
            mask = y_s == cls
            ax.scatter(X_s[mask, 0], X_s[mask, 1], color=colors[cls], alpha=0.6, s=15,
                       label=f"Classe {cls}")
            ax.scatter(mx, my, color=colors[cls], marker="X", s=150, edgecolor="black",
                       linewidth=1.2)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_title(f"s = {s}")
        ax.set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figura 2 — Nuvens para diferentes fatores de escala do desvio-padrão")
    return fig


def plot_figure3(mixing_rates):
    fig, ax = plt.subplots(figsize=(6, 5))
    scales_sorted = sorted(mixing_rates.keys())
    rates = [mixing_rates[s] for s in scales_sorted]
    ax.plot(scales_sorted, rates, marker="o", color="tab:red", label="Mixing rate")
    ax.set_title("Figura 3 — Taxa de mistura em função de s")
    ax.set_xlabel("Fator de escala s")
    ax.set_ylabel("Mixing rate")
    ax.legend()
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Item A ---
    X, y = generate_clouds(PARAMS, N_PER_CLASS, RNG)
    fig1 = plot_figure1(X, y, PARAMS)
    fig1.savefig(FIGURES / "fig01-point-clouds.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)  # (2)!

    # --- Item B ---
    datasets_b = {}
    for s in SCALES:
        params_s = scale_params(PARAMS, s)
        X_s, y_s = generate_clouds(params_s, N_PER_CLASS, RNG)
        datasets_b[s] = (X_s, y_s, params_s)

    fig2 = plot_figure2(datasets_b, SCALES)
    fig2.savefig(FIGURES / "fig02-scales.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    r_ij_s1 = separation_ratios(PARAMS)
    print("Separation ratio (s = 1):")
    for (ci, cj), r in r_ij_s1.items():
        print(f"  ({ci}, {cj}): r_ij = {r:.3f}")
    smallest_pair = min(r_ij_s1, key=r_ij_s1.get)
    smallest_r = r_ij_s1[smallest_pair]
    print(f"Menor r_ij: par {smallest_pair} = {smallest_r:.3f}")
    print(f"Em s = 2, esse par teria r_ij = {smallest_r / 2:.3f} (r_ij escala com 1/s)")

    mixing_rates = {}
    for s in SCALES:
        X_s, y_s, params_s = datasets_b[s]
        mixing_rates[s] = mixing_rate(X_s, y_s, params_s)
    print("Mixing rate por s:")
    for s, mr in mixing_rates.items():
        print(f"  s = {s}: mixing rate = {mr:.3f}")

    fig3 = plot_figure3(mixing_rates)
    fig3.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)


if __name__ == "__main__":
    main()
