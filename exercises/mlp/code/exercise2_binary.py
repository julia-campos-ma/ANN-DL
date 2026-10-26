"""Exercício 2 — Classificação binária que uma reta não resolve.

Gera os dados (item A, Figura 1), treina a linha de base linear (item B), treina o MLP
escrito à mão (item C, Figuras 2 e 3) e imprime os números usados na análise (item D).

A linha de base também é escrita à mão: regressão logística é exatamente o ``MLP`` com
``layer_sizes=[2, 1]`` (nenhuma camada oculta, saída sigmoide, perda BCE). O scikit-learn
é usado apenas para gerar e dividir os dados, como pede o enunciado.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/exercise2_binary.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from mlp import MLP, NAVY, ORANGE, PINK, gradient_check, standardize

FIGURES = Path(__file__).resolve().parents[1] / "figures"
rng = np.random.default_rng(42)  # (1)!

# Hiperparâmetros escolhidos (ver relatório, item C)
ARCH = [2, 8, 1]       # 2 entradas -> 8 tanh -> 1 sigmoide
LR = 0.5               # taxa de aprendizado η
EPOCHS = 2000          # épocas de descida do gradiente em lote completo
BASE_LR, BASE_EPOCHS = 0.5, 2000   # mesmos valores para a linha de base

COLORS = {0: NAVY, 1: PINK}


def make_data():
    """Item A: três blobs; o do centro é a classe 0 e os dois das pontas, a classe 1."""
    X, c = make_blobs(
        n_samples=[500, 250, 250],
        centers=[[0, 0], [3, 3], [-3, -3]],
        cluster_std=1.2,
        random_state=42,
    )
    y = (c > 0).astype(int)   # blob 0 -> classe 0; blobs 1 e 2 -> classe 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X, y, c, X_tr, X_te, y_tr, y_te


def scatter_classes(ax, X, y, s=12, alpha=0.55):
    for cls in (0, 1):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=s, alpha=alpha, color=COLORS[cls],
                   edgecolor="none", label=f"Classe {cls}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def plot_figure1(X, y):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 1 — 1000 pontos coloridos por y (classe 1 = dois blobs das pontas)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    return fig


def plot_figure2(model):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(model.loss_history) + 1), model.loss_history,
            color=PINK, lw=2, label=f"MLP {ARCH}, η = {LR}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de treino (entropia cruzada binária)")
    ax.set_title("Figura 2 — Perda de treino por época")
    ax.legend()
    return fig


def plot_figure3(model, scaler, X_te, y_te):
    """Fronteira de decisão: classifica uma grade fina do plano (padronizada com as
    estatísticas do TREINO) e pinta cada região com a cor da classe prevista."""
    mu, sd = scaler
    x1 = np.linspace(X_te[:, 0].min() - 1.5, X_te[:, 0].max() + 1.5, 400)
    x2 = np.linspace(X_te[:, 1].min() - 1.5, X_te[:, 1].max() + 1.5, 400)
    G1, G2 = np.meshgrid(x1, x2)
    grid = np.c_[G1.ravel(), G2.ravel()]
    P = model.predict_proba((grid - mu) / sd)[:, 0].reshape(G1.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(G1, G2, (P >= 0.5).astype(int), levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap([NAVY, PINK]), alpha=0.18)
    ax.contour(G1, G2, P, levels=[0.5], colors=ORANGE, linewidths=2)
    scatter_classes(ax, X_te, y_te, s=22, alpha=0.85)
    wrong = model.predict((X_te - mu) / sd) != y_te
    ax.scatter(X_te[wrong, 0], X_te[wrong, 1], marker="x", s=45, color=ORANGE,
               label=f"Erros ({wrong.sum()})")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=NAVY, alpha=0.25), Patch(color=PINK, alpha=0.25),
                Line2D([], [], color=ORANGE, lw=2)]
    labels += ["Região prevista: classe 0", "Região prevista: classe 1", "Fronteira (ŷ = 0,5)"]
    ax.legend(handles, labels, loc="upper left", fontsize=8.5)
    ax.set_title(f"Figura 3 — Fronteira de decisão do MLP sobre o teste "
                 f"(acurácia {1 - wrong.mean():.1%})")
    ax.set_aspect("equal")
    return fig


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)  # (2)!


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Item A ----------------------------------------------------------------
    X, y, c, X_tr, X_te, y_tr, y_te = make_data()
    print(f"treino: {len(y_tr)} pontos ({y_tr.mean():.0%} classe 1); "
          f"teste: {len(y_te)} pontos ({y_te.mean():.0%} classe 1)")
    print(f"média da classe 0 = {X[y == 0].mean(0).round(3)}, "
          f"média da classe 1 = {X[y == 1].mean(0).round(3)}")
    save(plot_figure1(X, y), "fig01-blobs.png")

    # Padronização com estatísticas do treino (o teste usa as mesmas mu e sd)
    Xs_tr, Xs_te, scaler = standardize(X_tr, X_te)
    Y_tr = y_tr[:, None].astype(float)           # (N, 1) para a saída sigmoide

    # --- Item B: linha de base = regressão logística = MLP sem camada oculta -----
    base = MLP([2, 1], output="sigmoid", rng=rng).fit(Xs_tr, Y_tr, BASE_LR, BASE_EPOCHS)
    print("\n== Linha de base (regressão logística, escrita à mão) ==")
    print(f"acurácia treino = {base.accuracy(Xs_tr, y_tr):.4f}, "
          f"teste = {base.accuracy(Xs_te, y_te):.4f}, perda final = {base.final_loss:.4f}")
    print(f"w = {base.W[0].ravel().round(4)}, b = {base.b[0].round(4)}")
    p_base = base.predict_proba(Xs_te)[:, 0]
    print(f"ŷ no teste: min = {p_base.min():.3f}, max = {p_base.max():.3f}; "
          f"fração prevista como 1 = {base.predict(Xs_te).mean():.3f}")

    # --- Item C: MLP do zero ------------------------------------------------------
    mlp = MLP(ARCH, output="sigmoid", rng=rng).fit(Xs_tr, Y_tr, LR, EPOCHS)
    print(f"\n== MLP {ARCH}, η = {LR}, {EPOCHS} épocas, {mlp.n_params} parâmetros ==")
    print(f"perda inicial = {mlp.loss_history[0]:.4f}, final = {mlp.final_loss:.4f}")
    print(f"acurácia treino = {mlp.accuracy(Xs_tr, y_tr):.4f}, "
          f"teste = {mlp.accuracy(Xs_te, y_te):.4f}")
    err = gradient_check(mlp, Xs_tr[:50], Y_tr[:50])
    print(f"checagem do gradiente (diferenças finitas, 50 amostras): erro relativo máx = {err:.2e}")

    save(plot_figure2(mlp), "fig02-loss-binary.png")
    save(plot_figure3(mlp, scaler, X_te, y_te), "fig03-decision-boundary.png")

    # --- Item D: números para a análise --------------------------------------------
    wrong = mlp.predict(Xs_te) != y_te
    r = np.linalg.norm(X_te, axis=1)
    print("\n== Diagnóstico (D) ==")
    print(f"erros do MLP no teste = {wrong.sum()}; distância média à origem dos erros = "
          f"{r[wrong].mean():.2f} (todos os pontos de teste: {r.mean():.2f})")
    proj = (X_te[:, 0] + X_te[:, 1]) / np.sqrt(2)  # coordenada ao longo da diagonal x1 = x2
    print(f"erros com |projeção na diagonal| entre 1 e 3: "
          f"{np.sum(wrong & (np.abs(proj) > 1) & (np.abs(proj) < 3))} de {wrong.sum()}")


if __name__ == "__main__":
    main()
