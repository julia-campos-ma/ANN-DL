"""
Step 2 -- the network: architecture, parameter count against the training rows, the
effect of the initialization on the activations, and the sanity check of overfitting a
tiny subset. Figure 2.

Run from the repository root:
    python docs/projects/classification/code/step2_network.py
"""

import numpy as np
import torch
from torch import nn

import common
import model as M
from common import SEED, section

plt = common.style()

CANDIDATES = [(64,), (128,), (128, 64), (256, 128), (256, 128, 64), (512, 256)]
BASE = (128, 64)


@torch.no_grad()
def activation_std(net, X):
    """Standard deviation of the output of every ReLU, for one forward pass."""
    stds, h = [], X
    for layer in net.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            stds.append(float(h.std()))
    return stds


def main():
    d = common.prepare()
    n_in, n_rows = d["X_fit"].shape[1], d["X_fit"].shape[0]

    section("Parameter count vs training rows")
    counts = {}
    for hidden in CANDIDATES:
        p = M.n_params(M.MLP(n_in, hidden))
        counts[str(hidden)] = p
        print(f"{' x '.join(map(str, hidden)):<16} params {p:>7,}   params / training row {p / n_rows:.3f}")
    base = M.MLP(n_in, BASE)
    print("\nbase network:", base)
    per_layer = [(n, tuple(p.shape), p.numel()) for n, p in base.named_parameters()]
    for n, s, k in per_layer:
        print(f"  {n:<14} {str(s):<12} {k:>6}")
    print(f"  total {M.n_params(base):,} parameters for {n_rows:,} training rows "
          f"({n_rows * n_in:,} input values)")

    section("Initialization: std of the ReLU outputs at init (5,000 training rows)")
    X = torch.as_tensor(d["X_fit"][:5000])
    print(f"input std (all {n_in} columns) {float(X.std()):.3f}")
    deep = (128,) * 8
    res = {}
    for name, init in (("PyTorch default", lambda m: m), ("He (kaiming_normal)", M.he_init)):
        for label, hidden in (("base 128-64", BASE), ("deep 8 x 128", deep)):
            stds = []
            for s in range(5):                               # average over 5 draws
                torch.manual_seed(SEED + s)
                stds.append(activation_std(init(M.MLP(n_in, hidden)), X))
            stds = np.mean(stds, axis=0)
            res[(name, label)] = stds
            print(f"{name:<20} {label:<13} " + " ".join(f"{v:.3f}" for v in stds))

    section("Sanity check: can the network overfit 256 training rows?")
    torch.manual_seed(SEED)
    Xs = torch.as_tensor(d["X_fit"][:256])
    ys = torch.tensor(d["y_fit"][:256], dtype=torch.float32)
    net = M.he_init(M.MLP(n_in, BASE))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(1, 1001):
        opt.zero_grad()
        loss = loss_fn(net(Xs), ys)
        loss.backward()
        opt.step()
        if step in (1, 100, 300, 1000):
            acc = float(((net(Xs) > 0).float() == ys).float().mean())
            print(f"step {step:4d}  loss {loss.item():.4f}  accuracy {acc:.3f}")

    # Figure 2 -- activation scale per layer
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, label in zip(axes, ("base 128-64", "deep 8 x 128")):
        for name, color in (("PyTorch default", "#888888"), ("He (kaiming_normal)", "#DD8452")):
            v = res[(name, label)]
            ax.plot(range(1, len(v) + 1), v, marker="o", color=color, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("camada oculta (saída da ReLU)")
        ax.set_ylabel("desvio-padrão da ativação")
        ax.set_xticks(range(1, len(res[("He (kaiming_normal)", label)]) + 1))
        ax.set_title(f"{'Rede base 128-64' if label.startswith('base') else 'Rede profunda 8 × 128'}")
        ax.legend()
    fig.suptitle("Figura 2 — Escala das ativações na inicialização: padrão do PyTorch × He",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig02-init.png")
    plt.close(fig)

    common.save_json("step2.json", {
        "n_in": n_in, "n_rows": n_rows, "param_counts": counts,
        "base_params": M.n_params(base),
        "act_std": {f"{a}|{b}": list(map(float, v)) for (a, b), v in res.items()},
    })


if __name__ == "__main__":
    main()
