"""VAE para o MNIST: dados, modelo, perda e laço de treino (PyTorch).

Este módulo é o "núcleo" usado pelos dois exercícios:

    * Exercício 1 — VAE com espaço latente de dimensão d = 16;
    * Exercício 2 — o MESMO modelo, os mesmos dados, a mesma semente, o mesmo β e o
      mesmo orçamento de treino, só que com d = 2.

As partes próprias do VAE são escritas à mão, como pede o enunciado: o truque da
reparametrização (``reparameterize``), os dois termos da perda (``vae_loss_terms``, com a
KL em forma fechada) e a amostragem da priori (feita nos scripts com ``torch.randn``).
O autograd do PyTorch calcula os gradientes; não usamos nenhuma classe pronta de VAE nem
``torch.distributions``.

Notação (a mesma da aula):

    q_φ(z | x) = N(μ_φ(x), diag σ²_φ(x))     codificador (encoder)
    p(z)       = N(0, I)                      priori
    p_θ(x | z) = Bernoulli(x̂_θ(z))            decodificador (decoder), x̂ ∈ (0, 1)^784

    −ELBO(x) = E_q[−log p_θ(x | z)]  +  β · KL(q_φ(z | x) ‖ p(z)),   β = 1
               └── reconstrução (BCE) ┘    └─────── KL ───────┘

Os dois termos são reportados SEPARADAMENTE, em nats por imagem: somados nos 784 pixels
(ou nas d dimensões, no caso da KL) e depois tirada a média sobre as imagens.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parents[4]   # raiz do repositório
DATA_DIR = REPO / "data"                     # o MNIST é baixado em data/MNIST (fora do git)
FIGURES = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"   # números em JSON

# Hiperparâmetros fixados pelo enunciado (item 1C)
SEED = 42
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
BETA = 1.0

# Paleta usada nas figuras (a mesma dos exercícios anteriores).
NAVY = "#173F5F"
PINK = "#D6336C"
LIGHT_BLUE = "#4DABF7"
ORANGE = "#E8590C"


# --------------------------------------------------------------------------------- dados
def load_mnist():
    """Item 1A: carrega o MNIST com pixels em [0, 1], achatados em vetores de 784.

    Devolve três ``TensorDataset`` (treino, validação, teste) com pares (x, rótulo).
    As 60 000 imagens oficiais de treino são divididas em 50 000 de treino e 10 000 de
    validação por ``random_split`` com um gerador de semente 42. As 10 000 imagens
    oficiais de teste formam o conjunto de teste e só são usadas nos números finais.
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)

    # data: uint8 (N, 28, 28) em {0..255}. Dividir por 255 dá exatamente o mesmo
    # resultado de transforms.ToTensor(), só que de uma vez para o conjunto inteiro.
    def to_dataset(ds):
        x = ds.data.reshape(-1, 784).float() / 255.0
        # Conferência: igual, pixel a pixel, ao ToTensor() aplicado imagem a imagem.
        assert torch.equal(transforms.ToTensor()(ds[0][0]).reshape(784), x[0])
        return TensorDataset(x, ds.targets.clone())

    full = to_dataset(train_full)
    train, val = random_split(full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(SEED))
    return train, val, to_dataset(test)


def stack(subset):
    """Junta um Dataset/Subset inteiro em dois tensores (X, y), na ordem original."""
    if isinstance(subset, TensorDataset):
        return subset.tensors
    x, y = subset.dataset.tensors
    idx = torch.as_tensor(subset.indices)
    return x[idx], y[idx]


# -------------------------------------------------------------------------------- modelo
class VAE(nn.Module):
    """Codificador 784 → 512 → 256 (ReLU), cabeças μ e log σ² (256 → d) e
    decodificador d → 256 → 512 → 784 (ReLU nas ocultas, sigmoide na saída)."""

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, latent_dim)       # μ_φ(x)
        self.logvar_head = nn.Linear(256, latent_dim)   # log σ²_φ(x)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),          # x̂ ∈ (0, 1): médias de Bernoulli
        )

    def encode(self, x):
        """Devolve os parâmetros da posteriori aproximada q_φ(z | x)."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        """Devolve x̂ = E[x | z], a probabilidade de cada pixel estar "aceso"."""
        return self.decoder(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar, generator=None):
    """Truque da reparametrização: z = μ + σ ⊙ ε, com ε ~ N(0, I).

    Toda a aleatoriedade fica em ε, que não depende dos parâmetros. Assim z é uma função
    determinística e diferenciável de (μ, log σ²): ∂z/∂μ = 1 e ∂z/∂log σ² = ½ σ ε, e o
    gradiente passa pela amostragem (KINGMA; WELLING, 2014). Prever log σ² em vez de σ
    deixa a cabeça livre em (−∞, +∞) e garante σ = exp(½ log σ²) > 0.
    """
    std = torch.exp(0.5 * logvar)                                         # σ
    eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)      # ε ~ N(0, I)
    return mu + std * eps


def vae_loss_terms(x, x_hat, mu, logvar):
    """Os dois termos da −ELBO, em nats por imagem (média sobre o lote).

    Reconstrução: −log p_θ(x | z) para um decodificador Bernoulli é a entropia cruzada
    binária SOMADA nos 784 pixels. ``reduction="sum"`` soma no lote todo; dividir pelo
    tamanho do lote dá a média por imagem. (Com ``reduction="mean"`` a BCE seria dividida
    também por 784, o que equivale a usar β ≈ 784 na KL.)

    KL em forma fechada entre N(μ, σ²) e N(0, 1), somada nas d dimensões latentes:

        KL = ½ Σ_j (μ_j² + σ_j² − 1 − log σ_j²)

    escrita termo a termo abaixo, sem ``torch.distributions``.
    """
    batch = x.shape[0]
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch
    kl_per_image = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
    kl = kl_per_image.mean()
    return recon, kl


# ------------------------------------------------------------------------------- treino
@torch.no_grad()
def evaluate(model, x, y=None, batch_size=1000, seed=123):
    """Reconstrução e KL médias (nats/imagem) num conjunto inteiro, sem atualizar pesos.

    O termo de reconstrução é estimado com UMA amostra z ~ q(z | x) por imagem (estimador
    não enviesado de E_q[−log p(x | z)]). As amostras de ε usam um gerador próprio, com
    semente fixa, para que a avaliação seja reprodutível e não consuma o gerador global
    usado no treino.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot_rec = tot_kl = 0.0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        x_hat, mu, logvar = model(xb, generator=gen)
        rec, kl = vae_loss_terms(xb, x_hat, mu, logvar)
        tot_rec += rec.item() * len(xb)
        tot_kl += kl.item() * len(xb)
    model.train()
    return tot_rec / len(x), tot_kl / len(x)


def train_vae(latent_dim, train_set, val_set, epochs=EPOCHS, verbose=True):
    """Itens 1C / 2A: treina um VAE e registra os dois termos por época.

    Adam (lr = 1e-3), lotes de 128, 20 épocas, β = 1. Em cada época guardamos:
      * no TREINO, a média dos termos ao longo da época (o modelo muda durante a época);
      * na VALIDAÇÃO, os termos com o modelo do fim da época.
    Também guardamos a validação ANTES do treino (época 0), usada na análise 1E.
    """
    torch.manual_seed(SEED)                                   # (semente antes do modelo)
    model = VAE(latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    x_val, _ = stack(val_set)

    history = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}
    history["val_epoch0"] = evaluate(model, x_val)

    for epoch in range(1, epochs + 1):
        sum_rec = sum_kl = 0.0
        n_seen = 0
        for xb, _ in loader:
            x_hat, mu, logvar = model(xb)                     # forward com z reparametrizado
            rec, kl = vae_loss_terms(xb, x_hat, mu, logvar)
            loss = rec + BETA * kl                            # −ELBO média do lote
            opt.zero_grad()
            loss.backward()                                   # autograd passa por z = μ + σε
            opt.step()
            sum_rec += rec.item() * len(xb)
            sum_kl += kl.item() * len(xb)
            n_seen += len(xb)

        val_rec, val_kl = evaluate(model, x_val)
        history["train_rec"].append(sum_rec / n_seen)
        history["train_kl"].append(sum_kl / n_seen)
        history["val_rec"].append(val_rec)
        history["val_kl"].append(val_kl)
        if verbose:
            print(f"d={latent_dim:2d} época {epoch:2d} | treino: rec {sum_rec / n_seen:7.2f} "
                  f"KL {sum_kl / n_seen:6.2f} | val: rec {val_rec:7.2f} KL {val_kl:6.2f}")
    model.eval()
    return model, history


def checkpoint_path(latent_dim):
    """Onde os scripts guardam o modelo treinado (pasta fora do git, ver .gitignore)."""
    return Path(__file__).resolve().parents[1] / "checkpoints" / f"vae_d{latent_dim}.pt"


def get_trained_vae(latent_dim, train_set, val_set):
    """Carrega o modelo salvo pelo script do exercício; se não existir, treina de novo.

    Como o treino é determinístico (sementes fixas), treinar de novo dá o mesmo modelo;
    o arquivo salvo só evita repetir o treino quando o Ex. 2 precisa do modelo d = 16.
    """
    path = checkpoint_path(latent_dim)
    if path.exists():
        model = VAE(latent_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model, None
    model, history = train_vae(latent_dim, train_set, val_set)
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model, history
