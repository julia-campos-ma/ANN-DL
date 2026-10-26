"""Exercício 4 — Uma segunda camada oculta ajuda?

Comparação controlada: mesmos dados, mesmo orçamento de treino (3000 épocas em lote
completo), mesma taxa de aprendizado (η = 0.5) e mesma semente do Exercício 3. Muda
apenas a profundidade: [4, 16, 3] (1 camada oculta) contra [4, 16, 16, 3] (2 camadas).

* Item A — arquitetura e acurácia da rede mais profunda (semente 42).
* Item B — Figura 6 (curvas de perda das duas redes nos mesmos eixos), tabela de
  comparação e média sobre 5 sementes de inicialização (42 a 46).

A semente 42 reproduz exatamente o Exercício 3: cada rede é criada com o seu próprio
``np.random.default_rng(seed)``, de modo que as duas partem "da mesma semente" e a
rede rasa com seed = 42 é idêntica à do Ex. 3.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/exercise4_depth.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from exercise3_multiclass import EPOCHS, LR, N_CLASSES, make_data
from mlp import MLP, NAVY, PINK, LIGHT_BLUE, one_hot, standardize

FIGURES = Path(__file__).resolve().parents[1] / "figures"

SHALLOW = [4, 16, 3]        # Exercício 3
DEEP = [4, 16, 16, 3]       # Exercício 4: só a profundidade muda
WIDE = [4, 50, 3]           # controle complementar: 1 camada com ~o mesmo nº de parâmetros
SEEDS = [42, 43, 44, 45, 46]


def train(arch, seed, Xs_tr, Y_tr):
    """Cria e treina uma rede com o seu próprio gerador de semente ``seed``."""
    return MLP(arch, output="softmax", rng=np.random.default_rng(seed)).fit(
        Xs_tr, Y_tr, LR, EPOCHS
    )


def plot_figure6(shallow, deep):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ep = np.arange(1, EPOCHS + 1)
    ax.plot(ep, shallow.loss_history, color=NAVY, lw=2,
            label=f"1 camada oculta {SHALLOW} ({shallow.n_params} parâm.)")
    ax.plot(ep, deep.loss_history, color=PINK, lw=2,
            label=f"2 camadas ocultas {DEEP} ({deep.n_params} parâm.)")
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de treino (CCE, escala log)")
    ax.set_title(f"Figura 6 — Perda de treino: rasa vs. profunda (η = {LR}, semente 42)")
    ax.legend()
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X_tr, X_te, y_tr, y_te = make_data()
    Xs_tr, Xs_te, _ = standardize(X_tr, X_te)
    Y_tr = one_hot(y_tr, N_CLASSES)

    # --- Item A + Figura 6: semente 42 -------------------------------------------
    shallow, deep = train(SHALLOW, 42, Xs_tr, Y_tr), train(DEEP, 42, Xs_tr, Y_tr)
    for name, m in [("rasa", shallow), ("profunda", deep)]:
        print(f"{name:9s} {m.layer_sizes}: {m.n_params} parâmetros, "
              f"perda final = {m.final_loss:.4f}, acurácia treino = "
              f"{m.accuracy(Xs_tr, y_tr):.4f}, teste = {m.accuracy(Xs_te, y_te):.4f}")
    fig = plot_figure6(shallow, deep)
    fig.savefig(FIGURES / "fig06-depth-comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Em que época a rede profunda alcança a perda final da rasa?
    reach = np.argmax(np.array(deep.loss_history) <= shallow.final_loss) + 1
    print(f"a profunda atinge a perda final da rasa ({shallow.final_loss:.4f}) na época {reach}")

    # --- Item B: várias sementes --------------------------------------------------
    print("\n== 5 sementes de inicialização (dados e split fixos) ==")
    rows = {a: {"loss": [], "tr": [], "te": []} for a in ("S", "D", "W")}
    for s in SEEDS:
        line = f"seed {s}:"
        for key, arch in (("S", SHALLOW), ("D", DEEP), ("W", WIDE)):
            m = train(arch, s, Xs_tr, Y_tr)
            rows[key]["loss"].append(m.final_loss)
            rows[key]["tr"].append(m.accuracy(Xs_tr, y_tr))
            rows[key]["te"].append(m.accuracy(Xs_te, y_te))
            line += f"  {arch}: perda {m.final_loss:.4f}, teste {rows[key]['te'][-1]:.4f}"
            if key == "S":   # par mais confundido da rede rasa (robustez do Ex. 3 D)
                cm = confusion_matrix(y_te, m.predict(Xs_te))
                pairs = {p: int(cm[p] + cm[p[::-1]]) for p in [(0, 1), (0, 2), (1, 2)]}
                line += f" [pares {pairs}]"
        print(line)

    print("\nmédia ± desvio-padrão sobre as sementes:")
    for key, arch in (("S", SHALLOW), ("D", DEEP), ("W", WIDE)):
        r = {k: np.array(v) for k, v in rows[key].items()}
        n_par = MLP(arch, output="softmax").n_params
        print(f"  {str(arch):14s} {n_par:4d} parâm. | perda final {r['loss'].mean():.4f} "
              f"± {r['loss'].std():.4f} | treino {r['tr'].mean():.4f} ± {r['tr'].std():.4f} "
              f"| teste {r['te'].mean():.4f} ± {r['te'].std():.4f}")
    diff = np.array(rows["D"]["te"]) - np.array(rows["S"]["te"])
    print(f"\ndiferença de acurácia no teste (profunda − rasa), por semente: {diff.round(4)}")
    print(f"média = {diff.mean():+.4f} ({diff.mean() * 100:+.2f} p.p.), "
          f"desvio-padrão = {diff.std():.4f}")
    gap_s = np.mean(rows["S"]["tr"]) - np.mean(rows["S"]["te"])
    gap_d = np.mean(rows["D"]["tr"]) - np.mean(rows["D"]["te"])
    print(f"gap treino − teste: rasa = {gap_s:.4f}, profunda = {gap_d:.4f}")


if __name__ == "__main__":
    main()
