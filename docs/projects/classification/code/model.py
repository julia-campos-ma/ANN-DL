"""
The network (Step 2) and the training loop (Step 3), in PyTorch.

- ``MLP``: fully connected ReLU network that returns ONE logit; the sigmoid lives inside
  ``BCEWithLogitsLoss`` during training and is applied explicitly only to predict.
- ``he_init``: Kaiming-normal weights (variance 2 / fan_in), zero biases.
- ``train_model``: mini-batch training with early stopping on the validation loss and
  restoration of the best weights.

Importing this module has no side effect; the scripts ``step2_*`` to ``step6_*`` use it.
"""

import copy
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

torch.set_num_threads(max(1, torch.get_num_threads()))


def set_seed(seed):
    """Fix every source of randomness that the training touches."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# --8<-- [start:mlp]
class MLP(nn.Module):
    """Input -> [Linear -> ReLU -> (Dropout)] x len(hidden) -> Linear(1) = logit."""

    def __init__(self, n_in, hidden=(128, 64), dropout=0.0):
        super().__init__()
        layers, width = [], n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))      # one output unit: the logit of "canceled"
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logits; no sigmoid here


def he_init(model):
    """He / Kaiming normal: Var(w) = 2 / fan_in for the ReLU layers, zero biases.

    PyTorch's default for nn.Linear is uniform with variance 1 / (3 fan_in), about a sixth
    of He, which shrinks the signal at every ReLU layer (Step 2 measures it). The output
    layer feeds a sigmoid, not a ReLU, so it gets the variance-preserving 1 / fan_in.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        if i < len(linears) - 1:
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.normal_(m.weight, mean=0.0, std=(1.0 / m.in_features) ** 0.5)
        nn.init.zeros_(m.bias)
    return model


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# --8<-- [end:mlp]


def make_optimizer(model, name="adam", lr=1e-3, weight_decay=0.0):
    """Adam without decay; AdamW (decoupled decay) when weight_decay > 0.

    Decay is applied to the weight matrices only, never to biases (Regularization class).
    """
    decay = [p for n, p in model.named_parameters() if n.endswith("weight")]
    no_decay = [p for n, p in model.named_parameters() if n.endswith("bias")]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    if weight_decay > 0 or name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.Adam(groups, lr=lr)


@torch.no_grad()
def predict_proba(model, X, batch=8192):
    model.eval()                                 # disables dropout
    X = torch.as_tensor(X)
    out = [torch.sigmoid(model(X[i:i + batch])) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


@torch.no_grad()
def bce(model, X, y, batch=8192):
    """Unweighted mean BCE: the same yardstick for every configuration."""
    model.eval()
    X, y = torch.as_tensor(X), torch.tensor(np.asarray(y), dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    total = sum(loss(model(X[i:i + batch]), y[i:i + batch]).item() for i in range(0, len(X), batch))
    return total / len(X)


# --8<-- [start:train]
def train_model(X_fit, y_fit, X_val, y_val, hidden=(128, 64), dropout=0.0, lr=1e-3,
                batch_size=256, weight_decay=0.0, pos_weight=1.0, optimizer="adam",
                max_epochs=100, patience=10, early_stopping=True, seed=42, verbose=False):
    """Train one MLP; returns (model with the best weights, history dict).

    Early stopping watches the unweighted validation BCE after every epoch; after
    ``patience`` epochs without improvement it stops and restores the weights of the best
    epoch. With ``early_stopping=False`` it runs ``max_epochs`` but still returns the best
    weights and the full curves (used by Step 4 to show what the stop avoids).
    """
    set_seed(seed)
    Xf = torch.as_tensor(X_fit)
    yf = torch.tensor(np.asarray(y_fit), dtype=torch.float32)
    model = he_init(MLP(Xf.shape[1], hidden, dropout))
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    gen = torch.Generator().manual_seed(seed)

    hist = {"train_loss": [], "val_loss": [], "val_ap": [], "train_loss_running": [], "seconds": []}
    best, best_epoch, best_state, wait = np.inf, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(Xf), generator=gen)
        running = 0.0
        for i in range(0, len(Xf), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        hist["train_loss_running"].append(running / len(Xf))
        hist["train_loss"].append(bce(model, X_fit, y_fit))     # eval mode, unweighted
        hist["val_loss"].append(bce(model, X_val, y_val))
        hist["val_ap"].append(float(average_precision_score(y_val, predict_proba(model, X_val))))
        hist["seconds"].append(time.perf_counter() - t0)
        if verbose:
            print(f"  epoch {epoch:3d}  train {hist['train_loss'][-1]:.4f}  "
                  f"val {hist['val_loss'][-1]:.4f}  val AP {hist['val_ap'][-1]:.4f}")
        if not np.isfinite(hist["val_loss"][-1]):
            break
        if hist["val_loss"][-1] < best - 1e-4:
            best, best_epoch, wait = hist["val_loss"][-1], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if early_stopping and wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)                       # restore the best epoch
    hist["best_epoch"] = best_epoch
    hist["stopped_epoch"] = epoch
    hist["best_val_loss"] = float(best)
    return model, hist
# --8<-- [end:train]
