"""Exercício 3 — Multiclasse com a MESMA rede.

Gera os dados (item A), treina o MLP com saída softmax + entropia cruzada categórica
(itens B e C, Figuras 4 e 5) e imprime os números da análise (item D).

A classe ``MLP`` é importada de ``mlp.py`` exatamente como no Exercício 2: as funções
``forward``, ``backward`` e ``update`` NÃO foram alteradas. Mudam apenas os argumentos
(``layer_sizes=[4, 16, 3]`` e ``output="softmax"``) e os hiperparâmetros.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/exercise3_multiclass.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP, PINK, gradient_check, one_hot, standardize

FIGURES = Path(__file__).resolve().parents[1] / "figures"
rng = np.random.default_rng(42)  # (1)!

ARCH = [4, 16, 3]      # 4 entradas -> 16 tanh -> 3 softmax
LR = 0.5               # taxa de aprendizado η (a mesma do Ex. 2)
EPOCHS = 3000          # épocas em lote completo
N_CLASSES = 3


def make_data():
    """Item A: 3 classes em 4 dimensões, cada classe formada por 2 grupos (clusters)."""
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
        n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def plot_figure4(model):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(model.loss_history) + 1), model.loss_history,
            color=PINK, lw=2, label=f"MLP {ARCH}, η = {LR}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de treino (entropia cruzada categórica)")
    ax.set_title("Figura 4 — Perda de treino por época (3 classes)")
    ax.legend()
    return fig


def plot_figure5(cm, acc):
    """Matriz de confusão (contagens) com a fração de cada linha anotada."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(cm, cmap="RdPu")
    row = cm / cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({row[i, j]:.0%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ticks = [f"Classe {k}" for k in range(cm.shape[0])]
    ax.set_xticks(range(cm.shape[1]), ticks)
    ax.set_yticks(range(cm.shape[0]), ticks)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Figura 5 — Matriz de confusão no teste\n(MLP {ARCH}, acurácia {acc:.1%})")
    fig.colorbar(im, ax=ax, label="Nº de amostras")
    return fig


def two_means(X, iters=50):
    """k-means com k = 2 escrito à mão (só para a análise do item D): devolve os
    2 centróides de um conjunto de pontos. A inicialização é DETERMINÍSTICA (não usa
    nenhum gerador aleatório, para respeitar a regra de um único ``rng`` no relatório):
    o 1º centróide é o ponto mais distante da média; o 2º, o mais distante do 1º."""
    c0 = X[np.argmax(((X - X.mean(axis=0)) ** 2).sum(axis=1))]
    c1 = X[np.argmax(((X - c0) ** 2).sum(axis=1))]
    C = np.array([c0, c1])
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (N, 2) distâncias²
        lab = d.argmin(axis=1)
        C = np.array([X[lab == k].mean(axis=0) for k in range(2)])
    return C


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)  # (2)!


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Item A ----------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = make_data()
    print(f"treino: {len(y_tr)}, teste: {len(y_te)}; "
          f"contagem por classe no teste: {np.bincount(y_te)}")
    Xs_tr, Xs_te, _ = standardize(X_tr, X_te)
    Y_tr = one_hot(y_tr, N_CLASSES)

    # --- Itens B e C: a MESMA classe MLP, agora com softmax + CCE ---------------
    mlp = MLP(ARCH, output="softmax", rng=rng).fit(Xs_tr, Y_tr, LR, EPOCHS)
    acc = mlp.accuracy(Xs_te, y_te)
    print(f"\n== MLP {ARCH}, η = {LR}, {EPOCHS} épocas, {mlp.n_params} parâmetros ==")
    print(f"perda inicial = {mlp.loss_history[0]:.4f}, final = {mlp.final_loss:.4f}")
    print(f"acurácia treino = {mlp.accuracy(Xs_tr, y_tr):.4f}, teste = {acc:.4f}")
    err = gradient_check(mlp, Xs_tr[:50], Y_tr[:50])
    print(f"checagem do gradiente (softmax + CCE, 50 amostras): erro relativo máx = {err:.2e}")

    save(plot_figure4(mlp), "fig04-loss-multiclass.png")
    cm = confusion_matrix(y_te, mlp.predict(Xs_te))
    save(plot_figure5(cm, acc), "fig05-confusion-matrix.png")
    print("matriz de confusão (linhas = verdadeira, colunas = prevista):\n", cm)

    # --- Item D: linha de base e estrutura de grupos ------------------------------
    # Regressão logística multinomial = MLP sem camada oculta ([4, 3], softmax).
    base = MLP([4, N_CLASSES], output="softmax", rng=rng).fit(Xs_tr, Y_tr, LR, EPOCHS)
    print(f"\n== Linha de base (regressão logística multinomial, à mão) ==")
    print(f"acurácia treino = {base.accuracy(Xs_tr, y_tr):.4f}, "
          f"teste = {base.accuracy(Xs_te, y_te):.4f}, perda final = {base.final_loss:.4f}")
    cm_base = confusion_matrix(y_te, base.predict(Xs_te))
    print("matriz de confusão da linha de base:\n", cm_base)

    # Pares de classes: erros nos dois sentidos
    print("\nerros por par de classes (MLP | linha de base):")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        print(f"  ({i}, {j}): {cm[i, j] + cm[j, i]:3d} | {cm_base[i, j] + cm_base[j, i]:3d}")

    # Estrutura de grupos: 2 centróides por classe (k-means à mão, no treino padronizado)
    cents = {k: two_means(Xs_tr[y_tr == k]) for k in range(N_CLASSES)}
    print("\ndistância entre centróides de grupos de classes diferentes (espaço padronizado):")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        D = np.linalg.norm(cents[i][:, None, :] - cents[j][None, :, :], axis=2)
        print(f"  ({i}, {j}): mínima = {D.min():.2f}, todas = {np.sort(D.ravel()).round(2)}")
    for k in range(N_CLASSES):
        print(f"  dentro da classe {k}: {np.linalg.norm(cents[k][0] - cents[k][1]):.2f}")
    print("\ncentróides dos grupos (padronizados):")
    for k in range(N_CLASSES):
        for c in cents[k]:
            print(f"  classe {k}: {c.round(2)}  sinais: {''.join('+' if v > 0 else '-' for v in c)}")


if __name__ == "__main__":
    main()
