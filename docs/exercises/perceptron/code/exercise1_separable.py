"""Exercise 1 — Perceptron em dados linearmente separáveis.

Gera as duas classes do item A (Figura 1), treina o perceptron com η = 0.01
(itens B e C, Figuras 2 e 3), repete o treino com η = 1.0 (item D.2) e verifica
numericamente o resultado algébrico do item D.3 (η só reescala os pesos).

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/exercise1_separable.py
"""

from pathlib import Path

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

MEAN0, MEAN1, VAR, N_PER_CLASS = [1.5, 1.5], [5.0, 5.0], 0.5, 1000
COLORS = {0: NAVY, 1: PINK}  # paleta: só tons de rosa, azul e laranja


def scatter_classes(ax, X, y):
    """Dispersão das duas classes (classe 0 azul-escuro, classe 1 rosa)."""
    for cls in (0, 1):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=12, alpha=0.5, color=COLORS[cls], label=f"Classe {cls}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def plot_figure1(X, y):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 1 — Dados separáveis: 1000 pontos por classe")
    ax.legend()
    return fig


def plot_figure2(X, y, p, p_big):
    """Fronteira w·x + b = 0 (η = 0.01) sobre os dados; erros marcados com X vermelho."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    wrong = p.predict(X) != y  # (2)!
    ax.scatter(X[wrong, 0], X[wrong, 1], marker="x", s=60, color=ORANGE,
               label=f"Erros ({wrong.sum()})")
    x1 = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    ax.plot(x1, decision_line(p.w, p.b, x1), color=ORANGE, lw=2, label="Fronteira (η = 0.01)")
    ax.plot(x1, decision_line(p_big.w, p_big.b, x1), color=LIGHT_BLUE, ls="--", lw=1.8,
            label="Fronteira (η = 1.0)")
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_title("Figura 2 — Fronteira de decisão do perceptron (dados separáveis)")
    ax.legend(loc="lower right")
    return fig


def plot_figure3(p):
    epochs = np.arange(1, len(p.acc_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, p.acc_history, marker="o", color=NAVY, label="Acurácia (η = 0.01)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no conjunto completo")
    ax.set_ylim(0.0, 1.05)  # eixo completo: as primeiras épocas oscilam muito
    ax.set_title("Figura 3 — Acurácia por época (dados separáveis)")
    ax.legend(loc="center right")
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Item A: dados -----------------------------------------------------
    X, y = generate_two_classes(MEAN0, MEAN1, VAR, N_PER_CLASS, RNG)
    print(f"X: {X.shape}, y: {y.shape}, por classe: {np.bincount(y)}")
    fig1 = plot_figure1(X, y)
    fig1.savefig(FIGURES / "fig01-separable-data.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)  # (3)!

    # --- Itens B e C: treino com η = 0.01 (o mesmo RNG segue o fluxo dos dados) ---
    p = Perceptron(lr=0.01, max_epochs=100, rng=RNG).fit(X, y)
    print("\n== η = 0.01 ==")
    print(f"w final = {p.w}, b final = {p.b:.4f}")
    print(f"épocas executadas = {p.epochs_run} (convergiu: {p.converged})")
    print(f"acurácia final = {p.accuracy(X, y):.4f}")
    print(f"atualizações por época: {p.updates_history}")
    print(f"w/||w|| = {p.w / np.linalg.norm(p.w)}")
    print(f"total de atualizações (erros corrigidos) = {p.total_updates}")

    # --- Item D.2: η = 1.0, mesma semente (novo RNG => mesma inicialização de w) ---
    rng2 = np.random.default_rng(42)
    X2, y2 = generate_two_classes(MEAN0, MEAN1, VAR, N_PER_CLASS, rng2)  # dados idênticos
    p_big = Perceptron(lr=1.0, max_epochs=100, rng=rng2).fit(X2, y2)
    print("\n== η = 1.0 ==")
    print(f"w final = {p_big.w}, b final = {p_big.b:.4f}")
    print(f"épocas executadas = {p_big.epochs_run} (convergiu: {p_big.converged})")
    print(f"acurácia final = {p_big.accuracy(X, y):.4f}")
    print(f"atualizações por época: {p_big.updates_history}")
    print(f"total de atualizações = {p_big.total_updates}")
    print(f"w/||w|| = {p_big.w / np.linalg.norm(p_big.w)}")
    u, v = p.w / np.linalg.norm(p.w), p_big.w / np.linalg.norm(p_big.w)
    cos = float(u @ v)
    print(f"cos(ângulo entre w/||w||) = {cos:.6f}  -> {np.degrees(np.arccos(min(cos, 1))):.3f}°")

    # --- Figuras 2 e 3 -----------------------------------------------------
    fig2 = plot_figure2(X, y, p, p_big)
    fig2.savefig(FIGURES / "fig02-decision-boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    fig3 = plot_figure3(p)
    fig3.savefig(FIGURES / "fig03-accuracy-epochs.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # --- Item D.3: verificação numérica partindo de w = 0, b = 0 ------------
    print("\n== D.3: partida em w = 0, b = 0 ==")
    runs = {}
    for lr in (0.01, 1.0):
        q = Perceptron(lr=lr, max_epochs=100, rng=np.random.default_rng(0), init_std=0.0).fit(X, y)
        runs[lr] = q
        print(f"η = {lr}: w = {q.w}, b = {q.b:.4f}, épocas = {q.epochs_run}, "
              f"acc = {q.accuracy(X, y):.4f}")
    a, b_ = runs[0.01], runs[1.0]
    ratio = 1.0 / 0.01
    print(f"w(η=1)/w(η=0.01) = {b_.w / a.w}  (esperado {ratio})")
    print(f"b(η=1)/b(η=0.01) = {b_.b / a.b:.4f}  (esperado {ratio})")
    print(f"mesmas previsões: {np.array_equal(a.predict(X), b_.predict(X))}; "
          f"mesmas épocas: {a.epochs_run == b_.epochs_run}")


if __name__ == "__main__":
    main()
