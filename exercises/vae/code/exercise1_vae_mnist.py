"""Exercício 1 — Um VAE no MNIST (d = 16).

Carrega e divide os dados (item A), treina o VAE do módulo ``vae.py`` (itens B e C,
Figura 1), gera reconstruções e amostras da priori (item D, Figuras 2 e 3) e imprime os
números usados na análise (item E). Os resultados numéricos também são gravados em
``results/results_ex1.json``.

Uso (a partir da raiz do repositório):

    python docs/exercises/vae/code/exercise1_vae_mnist.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from vae import (FIGURES, RESULTS, LIGHT_BLUE, NAVY, checkpoint_path, evaluate, load_mnist,
                 reparameterize, stack, train_vae, vae_loss_terms)

LATENT_DIM = 16
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def grid_image(images, nrows, ncols, pad=2):
    """Monta uma única imagem (nrows × ncols dígitos 28×28) com bordas brancas."""
    imgs = images.reshape(-1, 28, 28)
    canvas = np.ones((nrows * (28 + pad) + pad, ncols * (28 + pad) + pad))
    for k, img in enumerate(imgs):
        r, c = divmod(k, ncols)
        canvas[pad + r * (28 + pad): pad + r * (28 + pad) + 28,
               pad + c * (28 + pad): pad + c * (28 + pad) + 28] = 1.0 - img   # dígito escuro
    return canvas


# ------------------------------------------------------------------- A — dados e divisão
train_set, val_set, test_set = load_mnist()
x_test, y_test = stack(test_set)
print(f"treino: {len(train_set)}  validação: {len(val_set)}  teste: {len(test_set)}")

# ----------------------------------------------------------------- B e C — modelo e treino
model, hist = train_vae(LATENT_DIM, train_set, val_set)            # (1)!
checkpoint_path(LATENT_DIM).parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), checkpoint_path(LATENT_DIM))         # reutilizado no Ex. 2 D

# Teste: só agora, com o modelo final (nenhuma decisão de treino usou o teste).
test_rec, test_kl = evaluate(model, x_test)
print(f"TESTE d=16: reconstrução {test_rec:.2f}  KL {test_kl:.2f}  "
      f"−ELBO {test_rec + test_kl:.2f} nats/imagem")

epochs = np.arange(1, len(hist["train_rec"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, name in [(axes[0], "rec", "Reconstrução (BCE)"), (axes[1], "kl", "KL")]:
    ax.plot(epochs, hist[f"train_{key}"], "o-", color=NAVY, ms=4, label="treino (média da época)")
    ax.plot(epochs, hist[f"val_{key}"], "s--", color=LIGHT_BLUE, ms=4, label="validação (fim da época)")
    ax.set_title(f"{name} por época — VAE d = 16")
    ax.set_xlabel("Época")
    ax.set_ylabel("nats por imagem")
    ax.set_xticks(epochs[::2] if len(epochs) > 10 else epochs)
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig01-loss-terms.png", dpi=150)
plt.close(fig)                                                      # (2)!

# ----------------------------------------------------------- D — reconstruções e amostras
with torch.no_grad():
    # Figura 2: primeira imagem de teste de cada dígito 0–9, decodificada a partir de μ.
    first_idx = [int((y_test == d).nonzero()[0]) for d in range(10)]
    x_first = x_test[first_idx]
    mu_first, _ = model.encode(x_first)
    recon_first = model.decode(mu_first)                            # (3)!

    # Figura 3: 64 amostras da priori z ~ N(0, I), sorteadas depois de manual_seed(0).
    torch.manual_seed(0)
    z_prior = torch.randn(64, LATENT_DIM)
    samples = model.decode(z_prior)

fig, axes = plt.subplots(2, 10, figsize=(11, 2.8))
for d in range(10):
    for row, imgs in enumerate([x_first, recon_first]):
        ax = axes[row, d]
        ax.imshow(imgs[d].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, d].set_title(f"dígito {d}\n(idx {first_idx[d]})", fontsize=8)
    axes[1, d].set_xlabel(f"{d}", fontsize=8)
axes[0, 0].set_ylabel("original", fontsize=9)
axes[1, 0].set_ylabel("reconstrução\n(decodifica μ)", fontsize=9)
fig.suptitle("Reconstruções do VAE d = 16 — primeira imagem de teste de cada dígito")
fig.supxlabel("Dígito", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "fig02-reconstructions.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 6.6))
ax.imshow(grid_image(samples.numpy(), 8, 8), cmap="gray", vmin=0, vmax=1)
ax.set_title("64 amostras da priori z ~ N(0, I) decodificadas (d = 16, seed 0)")
ax.set_xlabel("coluna da grade (amostras 1–8)")
ax.set_ylabel("linha da grade (amostras 1–8)")
ticks = 2 + 30 * np.arange(8) + 14
ax.set_xticks(ticks, range(1, 9))
ax.set_yticks(ticks, range(1, 9))
fig.tight_layout()
fig.savefig(FIGURES / "fig03-prior-samples.png", dpi=150)
plt.close(fig)

# -------------------------------------------------------------- E — números para a análise
x_val, _ = stack(val_set)
x_train, _ = stack(train_set)
with torch.no_grad():
    mu_tr, logvar_tr = model.encode(x_train)
    mu_te, logvar_te = model.encode(x_test)

    # (E1) KL por dimensão latente na validação, e dimensões "ativas" (Var_x[μ_j] > 0,01;
    # BURDA; GROSSE; SALAKHUTDINOV, 2016).
    mu_va, logvar_va = model.encode(x_val)
    kl_dim = 0.5 * (mu_va.pow(2) + logvar_va.exp() - 1 - logvar_va).mean(0)
    active = int((mu_va.var(0) > 0.01).sum())

    # (E2a) Onde as amostras da priori caem. A posteriori AGREGADA é a mistura de todas as
    # posterioris de treino, q(z) = (1/N) Σ_i N(z; μ_i, σ_i²). Comparamos log q(z) e log p(z)
    # nas 64 amostras da Figura 3 e em 2000 z ~ q(z | x) de imagens de teste.
    std_tr = logvar_tr.mul(0.5).exp()
    active_mask = mu_tr.var(0) > 0.01
    LOG2PI = float(np.log(2 * np.pi))

    def log_q_agg(z):
        out = []
        for zb in z.split(500):
            dz = (zb[:, None, :] - mu_tr[None]) / std_tr[None]
            lp = (-0.5 * dz.pow(2) - std_tr.log()[None] - 0.5 * LOG2PI).sum(-1)
            out.append(torch.logsumexp(lp, 1) - np.log(len(mu_tr)))
        return torch.cat(out)

    def log_prior(z):
        return (-0.5 * z.pow(2) - 0.5 * LOG2PI).sum(-1)

    gen_q = torch.Generator().manual_seed(1)
    z_test = reparameterize(mu_te[:2000], logvar_te[:2000], gen_q)
    lq_prior, lq_test = log_q_agg(z_prior), log_q_agg(z_test)
    frac_holes = float((lq_prior < lq_test.quantile(0.05)).float().mean())
    # Distância ao μ de treino mais próximo, só nas dimensões ativas.
    def nn_dist(a):
        return torch.cdist(a[:, active_mask], mu_tr[:, active_mask]).min(1).values
    d_prior, d_test = nn_dist(z_prior), nn_dist(mu_te[:2000])
    # Variância por dimensão da posteriori agregada (= Var μ + E σ²) e σ médio.
    agg_var = mu_tr.var(0) + logvar_tr.exp().mean(0)
    sigma_active = float(std_tr[:, active_mask].mean())
    sigma_inactive = float(std_tr[:, ~active_mask].mean())

    # (E2b) Quão "cinzentas" são as saídas: fração de pixels com 0,1 < x̂ < 0,9.
    def gray_frac(imgs):
        return float(((imgs > 0.1) & (imgs < 0.9)).float().mean())

    recon_test_mu = model.decode(mu_te)
    gray_data = gray_frac(x_test)
    gray_recon = gray_frac(recon_test_mu)
    gray_prior = gray_frac(samples)

    # (E3) Decodificar μ ou um z sorteado: BCE média no teste (uma amostra por imagem).
    gen = torch.Generator().manual_seed(7)
    z_samp = reparameterize(mu_te, logvar_te, gen)
    bce_mu, _ = vae_loss_terms(x_test, recon_test_mu, mu_te, logvar_te)
    bce_z, _ = vae_loss_terms(x_test, model.decode(z_samp), mu_te, logvar_te)
    sigma_mean = float(logvar_te.mul(0.5).exp().mean())

results = {
    "test_rec": test_rec, "test_kl": test_kl, "test_neg_elbo": test_rec + test_kl,
    "val_kl_epoch0": hist["val_epoch0"][1], "val_rec_epoch0": hist["val_epoch0"][0],
    "val_kl_first": hist["val_kl"][0], "val_kl_last": hist["val_kl"][-1],
    "history": {k: v for k, v in hist.items() if k != "val_epoch0"},
    "first_idx": first_idx,
    "kl_per_dim_val": kl_dim.tolist(), "active_dims": active,
    "log_q_agg_median": {"prior": float(lq_prior.median()), "test": float(lq_test.median())},
    "log_prior_median": {"prior": float(log_prior(z_prior).median()),
                         "test": float(log_prior(z_test).median())},
    "frac_prior_below_test_p5": frac_holes,
    "nn_dist_active_median": {"prior": float(d_prior.median()), "test": float(d_test.median())},
    "agg_var_per_dim": agg_var.tolist(),
    "sigma_mean_active": sigma_active, "sigma_mean_inactive": sigma_inactive,
    "gray_frac": {"data": gray_data, "recon_mu": gray_recon, "prior": gray_prior},
    "bce_decode_mu": float(bce_mu), "bce_decode_z": float(bce_z), "sigma_mean": sigma_mean,
}
(RESULTS / "results_ex1.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
