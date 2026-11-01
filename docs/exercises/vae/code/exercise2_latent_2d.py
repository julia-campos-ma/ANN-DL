"""Exercício 2 — Um espaço latente que dá para ver (d = 2).

Treina o mesmo VAE do Exercício 1 com d = 2 (item A), desenha o conjunto de teste
codificado (item B, Figura 4), decodifica uma grade no plano latente (item C, Figura 5) e
compara interpolações no espaço dos pixels e no espaço latente do modelo d = 16 (item D,
Figura 6). Imprime e grava em ``results/results_ex2.json`` os números da análise (item E).

O modelo d = 16 vem de ``checkpoints/vae_d16.pt``, salvo pelo script do Exercício 1; se o
arquivo não existir, ele é treinado de novo com a mesma semente (dá o mesmo modelo).

Uso (a partir da raiz do repositório):

    python docs/exercises/vae/code/exercise2_latent_2d.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from vae import (FIGURES, RESULTS, evaluate, get_trained_vae, load_mnist, stack,
                 train_vae)

FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
GRID_N = 15                       # grade 15 × 15 (item C)
IDX_A, IDX_B = 3, 2               # (1)!
N_STEPS = 10                      # t = 0, 1/9, ..., 1 (item D)

train_set, val_set, test_set = load_mnist()
x_train, y_train = stack(train_set)
x_test, y_test = stack(test_set)

# --------------------------------------------------------------------- A — VAE com d = 2
model2, hist2 = train_vae(2, train_set, val_set)       # mesma semente, β, dados e orçamento
model16, _ = get_trained_vae(16, train_set, val_set)

test_rec2, test_kl2 = evaluate(model2, x_test)
test_rec16, test_kl16 = evaluate(model16, x_test)
print(f"TESTE d=2 : reconstrução {test_rec2:.2f}  KL {test_kl2:.2f}  −ELBO {test_rec2 + test_kl2:.2f}")
print(f"TESTE d=16: reconstrução {test_rec16:.2f}  KL {test_kl16:.2f}  −ELBO {test_rec16 + test_kl16:.2f}")

with torch.no_grad():
    mu_te, logvar_te = model2.encode(x_test)
    mu_tr, _ = model2.encode(x_train)
mu_np = mu_te.numpy()
labels = y_test.numpy()
cmap = plt.get_cmap("tab10")

# --------------------------------------------------------- B — conjunto de teste codificado
fig, ax = plt.subplots(figsize=(7.5, 6.5))
# Desenha os pontos numa ordem embaralhada (fixa): assim nenhuma classe fica sempre por cima.
order = np.random.default_rng(0).permutation(len(mu_np))
ax.scatter(mu_np[order, 0], mu_np[order, 1], c=labels[order], cmap=cmap, vmin=-0.5,
                vmax=9.5, s=4, alpha=0.6, linewidths=0)
handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(d), ms=6, label=str(d))
           for d in range(10)]
ax.legend(handles=handles, title="dígito", loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.set_title("μ das 10 000 imagens de teste no plano latente (VAE d = 2)")
ax.set_xlabel("$\\mu_1$")
ax.set_ylabel("$\\mu_2$")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "fig04-latent-scatter.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ C — plano decodificado
q = np.linspace(0.05, 0.95, GRID_N)              # 15 quantis igualmente espaçados
zs = norm.ppf(q)                                 # z = Φ⁻¹(q): mesma massa da priori por faixa
grid = np.array([[zs[j], zs[i]] for i in range(GRID_N - 1, -1, -1) for j in range(GRID_N)])
with torch.no_grad():                            # linha 0 = maior z₂ (topo da figura)
    decoded = model2.decode(torch.tensor(grid, dtype=torch.float32)).numpy()

pad = 2
canvas = np.ones((GRID_N * (28 + pad) + pad,) * 2)
for k, img in enumerate(decoded):
    r, c = divmod(k, GRID_N)
    canvas[pad + r * 30: pad + r * 30 + 28, pad + c * 30: pad + c * 30 + 28] = 1 - img.reshape(28, 28)
fig, ax = plt.subplots(figsize=(8.5, 8.8))
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
centers = pad + 30 * np.arange(GRID_N) + 14
ax.set_xticks(centers, [f"{v:.2f}" for v in zs], rotation=90, fontsize=8)
ax.set_yticks(centers, [f"{v:.2f}" for v in zs[::-1]], fontsize=8)
ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
ax.set_title("Plano latente decodificado (d = 2): grade 15 × 15 em $z = \\Phi^{-1}(q)$, "
             "$q \\in [0{,}05;\\ 0{,}95]$")
fig.tight_layout()
fig.savefig(FIGURES / "fig05-latent-grid.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------------ D — interpolação
t = torch.linspace(0, 1, N_STEPS)
xa, xb = x_test[IDX_A], x_test[IDX_B]
pixel_path = (1 - t)[:, None] * xa + t[:, None] * xb             # (1 − t)·x_a + t·x_b
with torch.no_grad():
    mu16, _ = model16.encode(torch.stack([xa, xb]))              # μ_a, μ_b do modelo d = 16
    latent_path = model16.decode((1 - t)[:, None] * mu16[0] + t[:, None] * mu16[1])

fig, axes = plt.subplots(2, N_STEPS, figsize=(12, 3.2))
for k in range(N_STEPS):
    for row, imgs in enumerate([pixel_path, latent_path]):
        ax = axes[row, k]
        ax.imshow(imgs[k].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, k].set_xlabel(f"t = {t[k]:.2f}", fontsize=8)
axes[0, 0].set_ylabel("pixels\n$(1-t)x_a + t x_b$", fontsize=9)
axes[1, 0].set_ylabel("latente d = 16\n$D((1-t)\\mu_a + t\\mu_b)$", fontsize=9)
fig.suptitle(f"Interpolação entre o teste #{IDX_A} (dígito {int(y_test[IDX_A])}) "
             f"e o teste #{IDX_B} (dígito {int(y_test[IDX_B])})")
fig.supxlabel("passo da interpolação t", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "fig06-interpolation.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------- E — números para a análise
# (E1) Quais dígitos se sobrepõem no plano: k-NN (k = 15) escrito à mão, com os μ de TREINO
# como referência, aplicado aos μ de teste. A matriz de confusão mostra que pares dividem a
# mesma região. (Só análise: o VAE não usa rótulos.)
def knn_predict(points, ref, ref_labels, k=15):
    """Rótulo mais frequente entre os k vizinhos mais próximos (distância euclidiana)."""
    out = []
    for pb in torch.as_tensor(points, dtype=torch.float32).split(1000):
        nbr = torch.cdist(pb, ref).topk(k, largest=False).indices        # (B, k)
        votes = torch.zeros(len(pb), 10).scatter_add_(1, ref_labels[nbr],
                                                       torch.ones(len(pb), k))
        out.append(votes.argmax(1))
    return torch.cat(out).numpy()


pred = knn_predict(mu_np, mu_tr, y_train)
conf = np.zeros((10, 10), dtype=int)
for a, b in zip(labels, pred):
    conf[a, b] += 1
pairs = sorted(((conf[i, j] + conf[j, i], i, j) for i in range(10) for j in range(i + 1, 10)),
               reverse=True)
# Rótulo "dominante" de cada célula da Figura 5 (mesmo k-NN), para localizar as transições.
cell_labels = knn_predict(grid, mu_tr, y_train).reshape(GRID_N, GRID_N)

# (E2) O que d = 2 descarta: variação DENTRO de cada classe que sobrevive à reconstrução,
# como fração da variação dentro da classe nos dados (média nas 10 classes).
with torch.no_grad():
    rec2 = model2.decode(mu_te)
    rec16 = model16.decode(model16.encode(x_test)[0])


def within_class_var(imgs):
    return float(np.mean([imgs[labels == d].var(0).sum() for d in range(10)]))


wcv = {"data": within_class_var(x_test.numpy()), "d2": within_class_var(rec2.numpy()),
       "d16": within_class_var(rec16.numpy())}

# (E3) Quanto de cada imagem do caminho é "meia-tinta": fração de pixels com 0,1 < x < 0,9.
# Um dígito real tem traços quase binários (≈ 9 % de pixels nessa faixa no teste).
def gray_frac(imgs):
    return ((imgs > 0.1) & (imgs < 0.9)).float().mean(1).tolist()


gray_path = {"pixel": gray_frac(pixel_path), "latent": gray_frac(latent_path),
             "test_data": float(((x_test > 0.1) & (x_test < 0.9)).float().mean())}

results = {
    "test_rec_d2": test_rec2, "test_kl_d2": test_kl2, "test_neg_elbo_d2": test_rec2 + test_kl2,
    "test_rec_d16": test_rec16, "test_kl_d16": test_kl16,
    "val_kl_first_d2": hist2["val_kl"][0], "val_kl_last_d2": hist2["val_kl"][-1],
    "history_d2": {k: v for k, v in hist2.items() if k != "val_epoch0"},
    "grid_z": zs.tolist(),
    "idx_a": IDX_A, "idx_b": IDX_B, "digit_a": int(y_test[IDX_A]), "digit_b": int(y_test[IDX_B]),
    "knn_test_acc": float((pred == labels).mean()),
    "top_confused_pairs": [[int(n), int(i), int(j)] for n, i, j in pairs[:8]],
    "confusion": conf.tolist(),
    "cell_labels_top_to_bottom": cell_labels.tolist(),
    "class_centroids": [mu_np[labels == d].mean(0).tolist() for d in range(10)],
    "within_class_var": wcv,
    "gray_frac_path": gray_path,
}
(RESULTS / "results_ex2.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: v for k, v in results.items() if k not in ("history_d2", "confusion")},
                 indent=1))
