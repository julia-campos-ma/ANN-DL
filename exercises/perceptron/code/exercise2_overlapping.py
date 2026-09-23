"""Exercise 2 — Perceptron em dados sobrepostos (algoritmo do bolso / pocket).

Gera as duas classes com sobreposição (item A, Figura 4), treina com a MESMA classe
``Perceptron`` do Exercise 1 (item B, sem nenhuma alteração), e desenha as Figuras 5
e 6 (item C). Ao final, imprime os números usados na análise (item D).

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/exercise2_overlapping.py
"""

from pathlib import Path
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np

from perceptron import (
    LIGHT_BLUE,
    NAVY,
    ORANGE,
    PINK,
    Perceptron,
    decision_line,
    generate_two_classes,
)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

MEAN0, MEAN1, VAR, N_PER_CLASS = [3.0, 3.0], [4.0, 4.0], 1.5, 1000
COLORS = {0: NAVY, 1: PINK}  # paleta: só tons de rosa, azul e laranja


def scatter_classes(ax, X, y):
    for cls in (0, 1):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=10, alpha=0.4, color=COLORS[cls], label=f"Classe {cls}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def plot_figure4(X, y):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 4 — Dados sobrepostos: 1000 pontos por classe")
    ax.legend()
    return fig


def plot_figure5(X, y, p):
    """Dois painéis: pesos finais (esq.) e pesos do bolso (dir.), cada um com a sua
    fronteira e os seus erros marcados com X vermelho. Eixos compartilhados."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    x1 = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    cases = [
        ("Pesos finais", p.w, p.b, ORANGE),
        ("Pesos do bolso (pocket)", p.pocket_w, p.pocket_b, LIGHT_BLUE),
    ]
    for ax, (name, w, b, color) in zip(axes, cases):
        scatter_classes(ax, X, y)
        wrong = p.predict(X, w, b) != y
        ax.scatter(X[wrong, 0], X[wrong, 1], marker="x", s=25, color=ORANGE, alpha=0.85,
                   label=f"Erros ({wrong.sum()})")
        ax.plot(x1, decision_line(w, b, x1), color=color, lw=2.5, label="Fronteira")
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        ax.set_title(f"{name} — acurácia {1 - wrong.mean():.2%}")
        ax.legend(loc="upper right")
    fig.suptitle("Figura 5 — Fronteiras de decisão em dados sobrepostos")
    return fig


def plot_figure6(p):
    epochs = np.arange(1, len(p.acc_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, p.acc_history, color=LIGHT_BLUE, lw=1.2, label="Acurácia atual (fim da época)")
    ax.plot(epochs, p.best_acc_history, color=ORANGE, lw=2.2, label="Melhor até agora (bolso)")
    ax.axhline(0.5, color=NAVY, ls=":", label="Chute (50%)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no conjunto completo")
    ax.set_title("Figura 6 — Acurácia atual vs. melhor acurácia (bolso)")
    ax.legend(loc="center right")
    return fig


def normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Item A ------------------------------------------------------------
    X, y = generate_two_classes(MEAN0, MEAN1, VAR, N_PER_CLASS, RNG)
    fig4 = plot_figure4(X, y)
    fig4.savefig(FIGURES / "fig04-overlapping-data.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)  # (2)!

    # --- Item B: mesma classe do Exercise 1, η = 0.01, 100 épocas ------------
    p = Perceptron(lr=0.01, max_epochs=100, rng=RNG).fit(X, y)
    print("== Pesos finais ==")
    print(f"w = {p.w}, b = {p.b:.4f}, acurácia = {p.accuracy(X, y):.4f}")
    print("== Pesos do bolso ==")
    print(f"w = {p.pocket_w}, b = {p.pocket_b:.4f}, acurácia = {p.pocket_acc:.4f}")
    print(f"bolso salvo na época {p.pocket_epoch} (atualização nº {p.pocket_update} "
          f"de {p.total_updates})")
    print(f"épocas executadas = {p.epochs_run} (convergiu: {p.converged})")

    # --- Item C ------------------------------------------------------------
    fig5 = plot_figure5(X, y, p)
    fig5.savefig(FIGURES / "fig05-boundaries.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    fig6 = plot_figure6(p)
    fig6.savefig(FIGURES / "fig06-accuracy-pocket.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)

    # --- Item D: números usados na análise -----------------------------------
    print("\n== Diagnóstico (D.1) ==")
    norms = np.linalg.norm(X, axis=1)
    print(f"||x||: média = {norms.mean():.2f}, mediana = {np.median(norms):.2f}")
    print(f"atualizações/época: média = {np.mean(p.updates_history):.2f}, "
          f"min = {min(p.updates_history)}, max = {max(p.updates_history)}")
    print(f"||w|| inicial = {np.linalg.norm(p.w_history[0]):.4f}, "
          f"||w|| final = {np.linalg.norm(p.w):.4f}")
    for name, w, b in [("final", p.w, p.b), ("bolso", p.pocket_w, p.pocket_b)]:
        preds = p.predict(X, w, b)
        d0 = (w @ np.array(MEAN0) + b) / np.linalg.norm(w)   # distância com sinal
        d1 = (w @ np.array(MEAN1) + b) / np.linalg.norm(w)
        print(f"{name}: fração prevista como classe 1 = {preds.mean():.3f}; "
              f"dist. com sinal à média0 = {d0:.2f}, à média1 = {d1:.2f}; "
              f"acurácia classe0 = {np.mean(preds[:N_PER_CLASS] == 0):.3f}, "
              f"classe1 = {np.mean(preds[N_PER_CLASS:] == 1):.3f}")

    # Teto teórico de um classificador linear: fronteira x1 + x2 = 7 (mediatriz das médias)
    d = np.linalg.norm(np.array(MEAN1) - np.array(MEAN0))
    acc_bayes = normal_cdf(d / (2 * sqrt(VAR)))
    print(f"\nAcurácia ótima teórica (Bayes, fronteira x1 + x2 = 7): {acc_bayes:.4f}")
    acc_opt = np.mean(((X[:, 0] + X[:, 1] - 7.0) >= 0).astype(int) == y)
    print(f"Acurácia empírica dessa reta nesta amostra: {acc_opt:.4f}")

    # --- Item D.3: verificação complementar (não substitui o argumento) -------
    print("\n== D.3: mais épocas / outros η (verificação complementar) ==")
    for lr, epochs in [(0.01, 100), (0.01, 1000), (0.001, 100), (0.1, 100), (1.0, 100)]:
        rng_i = np.random.default_rng(42)
        Xi, yi = generate_two_classes(MEAN0, MEAN1, VAR, N_PER_CLASS, rng_i)
        q = Perceptron(lr=lr, max_epochs=epochs, rng=rng_i).fit(Xi, yi)
        print(f"η = {lr:<6} épocas = {epochs:<5} final = {q.accuracy(Xi, yi):.4f}  "
              f"bolso = {q.pocket_acc:.4f} (época {q.pocket_epoch})  convergiu = {q.converged}")


if __name__ == "__main__":
    main()
