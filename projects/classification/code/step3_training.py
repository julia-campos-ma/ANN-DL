"""
Step 3 -- training: learning-rate sweep for Adam and for SGD with Nesterov momentum,
batch-size comparison, and the reference training run with early stopping on validation
(best weights restored). Figures 3 and 4.

Run from the repository root:
    python docs/projects/classification/code/step3_training.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import common
import model as M
from common import section

plt = common.style()

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SWEEP_EPOCHS = 5
BASE = dict(hidden=(128, 64), dropout=0.0, weight_decay=0.0, pos_weight=1.0)
LR, BATCH = 3e-4, 256          # chosen from the sweep; the reasons are in the report
MAX_EPOCHS, PATIENCE = 100, 10


def main():
    d = common.prepare()
    Xf, yf, Xv, yv = d["X_fit"], d["y_fit"], d["X_val"], d["y_val"]
    out = {}

    section(f"Learning-rate sweep ({SWEEP_EPOCHS} epochs, batch {BATCH}, 128-64, no regularization)")
    sweep = {}
    for opt in ("adam", "sgd"):
        for lr in LRS:
            _, h = M.train_model(Xf, yf, Xv, yv, lr=lr, batch_size=BATCH, optimizer=opt,
                                 max_epochs=SWEEP_EPOCHS, early_stopping=False, **BASE)
            sweep[(opt, lr)] = h
            print(f"{opt:<5} lr={lr:<7} train loss/epoch " + " ".join(f"{v:.3f}" for v in h["train_loss"])
                  + f" | val loss min {min(h['val_loss']):.4f} (epoch {int(np.argmin(h['val_loss'])) + 1})"
                  f" | val AP max {max(h['val_ap']):.4f}")
    out["sweep"] = {f"{o}|{lr}": {k: h[k] for k in ("train_loss", "val_loss", "val_ap")}
                    for (o, lr), h in sweep.items()}

    section("Batch size (Adam, lr scaled by the square-root rule from 3e-4 at 256)")
    batch = {}
    for bs in (64, 256, 1024):
        lr = LR * np.sqrt(bs / 256)
        _, h = M.train_model(Xf, yf, Xv, yv, lr=lr, batch_size=bs, max_epochs=MAX_EPOCHS,
                             patience=PATIENCE, **BASE)
        b = h["best_epoch"]
        batch[bs] = {"lr": lr, "best_epoch": b, "stopped": h["stopped_epoch"], "val_loss": h["best_val_loss"],
                     "val_ap": h["val_ap"][b - 1], "sec_per_epoch": float(np.mean(h["seconds"])),
                     "steps_per_epoch": int(np.ceil(len(Xf) / bs))}
        print(f"batch {bs:<5} lr {lr:.2e}  steps/epoch {batch[bs]['steps_per_epoch']:<5} "
              f"best epoch {b:<3} stopped {h['stopped_epoch']:<3} val loss {h['best_val_loss']:.4f}  "
              f"val AP {h['val_ap'][b - 1]:.4f}  {batch[bs]['sec_per_epoch']:.2f} s/epoch")
    out["batch"] = batch

    section(f"Reference run: Adam lr {LR}, batch {BATCH}, up to {MAX_EPOCHS} epochs, patience {PATIENCE}")
    net, h = M.train_model(Xf, yf, Xv, yv, lr=LR, batch_size=BATCH, max_epochs=MAX_EPOCHS,
                           patience=PATIENCE, verbose=True, **BASE)
    b = h["best_epoch"]
    print(f"best epoch {b}, stopped at {h['stopped_epoch']}; val loss {h['best_val_loss']:.4f}, "
          f"val AP at best {h['val_ap'][b - 1]:.4f}; train loss at best {h['train_loss'][b - 1]:.4f}, "
          f"at stop {h['train_loss'][-1]:.4f}; val loss at stop {h['val_loss'][-1]:.4f}")
    lr_model = LogisticRegression(C=common.load_json("step1.json")["logistic"]["C"], max_iter=5000).fit(Xf, yf)
    ll_fit = log_loss(yf, lr_model.predict_proba(Xf)[:, 1])
    ll_val = log_loss(yv, lr_model.predict_proba(Xv)[:, 1])
    print(f"logistic regression for reference: train loss {ll_fit:.4f}, val loss {ll_val:.4f}, "
          f"gap {ll_val - ll_fit:.4f}; MLP gap at its best epoch {h['val_loss'][b - 1] - h['train_loss'][b - 1]:.4f}")
    out["reference"] = {k: h[k] for k in ("train_loss", "val_loss", "val_ap", "best_epoch", "stopped_epoch",
                                          "best_val_loss")}
    out["logistic_loss"] = {"fit": ll_fit, "val": ll_val}

    # Figure 3 -- learning-rate sweep
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, opt, title in zip(axes, ("adam", "sgd"), ("Adam", "SGD + Nesterov (momento 0,9)")):
        for i, lr in enumerate(LRS):
            v = sweep[(opt, lr)]["val_loss"]
            ax.plot(range(1, len(v) + 1), v, marker="o", ms=3, color=cmap(i / (len(LRS) - 1)), label=f"lr = {lr:g}")
        ax.axhline(ll_val, color="black", ls=":", lw=1, label="logística (validação)")
        ax.set_xlabel("época")
        ax.set_title(title)
        ax.set_ylim(0.38, 0.70)
        ax.set_xticks(range(1, SWEEP_EPOCHS + 1))
    axes[0].set_ylabel("perda de validação (BCE)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=8, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.suptitle("Figura 3 — Varredura de learning rate: perda de validação nas 5 primeiras épocas",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig03-lr-sweep.png")
    plt.close(fig)

    # Figure 4 -- training curves of the reference run
    ep = np.arange(1, len(h["val_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(ep, h["train_loss"], color=common.COLORS[0], marker="o", ms=3, label="treino")
    a1.plot(ep, h["val_loss"], color=common.COLORS[1], marker="o", ms=3, label="validação")
    a1.axhline(ll_fit, color=common.COLORS[0], ls=":", lw=1, label="logística (treino)")
    a1.axhline(ll_val, color=common.COLORS[1], ls=":", lw=1, label="logística (validação)")
    a1.axvline(b, color="black", ls="--", lw=1, label=f"melhor época ({b})")
    a1.set_xlabel("época")
    a1.set_ylabel("perda (BCE média)")
    a1.set_title("Perda de treino e de validação")
    a1.legend()
    a2.plot(ep, h["val_ap"], color=common.COLORS[1], marker="o", ms=3, label="MLP (validação)")
    a2.axhline(common.load_json("step1.json")["logistic"]["val"]["average_precision"], color="black", ls=":", lw=1,
               label="logística (validação)")
    a2.axvline(b, color="black", ls="--", lw=1)
    a2.set_xlabel("época")
    a2.set_ylabel("average precision")
    a2.set_title("AP de validação")
    a2.legend()
    from matplotlib.ticker import MaxNLocator
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Figura 4 — Curvas de treino da rede de referência (128-64, Adam, lr {LR:g}, batch {BATCH})",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig04-curves.png")
    plt.close(fig)

    common.save_json("step3.json", out)


if __name__ == "__main__":
    main()
