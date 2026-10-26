"""MLP escrito à mão (só NumPy): ativações, perdas, passagem direta, gradientes e atualização.

Este módulo é o "núcleo" usado SEM NENHUMA ALTERAÇÃO pelos Exercícios 2, 3 e 4:

    * Exercício 2 — saída com 1 unidade + sigmoide + entropia cruzada binária;
    * Exercício 3 — saída com 3 unidades + softmax + entropia cruzada categórica;
    * Exercício 4 — mesma rede do Ex. 3, com duas camadas ocultas.

O que muda entre os exercícios são só os ARGUMENTOS do construtor (``layer_sizes`` e
``output``) e os hiperparâmetros do ``fit``. Isso funciona porque, para os dois pares
"ativação de saída + perda" usados, o gradiente da perda em relação à pré-ativação da
saída tem a mesma forma (derivação no relatório, Ex. 3 B):

    sigmoide + BCE   ->  dL/du = (ŷ − y) / N
    softmax  + CCE   ->  dL/du = (ŷ − y) / N      (y em one-hot)

Notação (igual à do Exercício 1): camada l tem pesos W^(l) (fan_in × fan_out) e viés
b^(l); u^(l) = h^(l−1) W^(l) + b^(l) é a pré-ativação e h^(l) = f(u^(l)) a ativação,
com h^(0) = X. As camadas ocultas usam tanh. As amostras ficam nas LINHAS de X.
"""

import numpy as np

# Paleta usada em todas as figuras (a mesma dos exercícios anteriores).
NAVY = "#173F5F"        # azul escuro — classe 0
PINK = "#D6336C"        # rosa — classe 1
LIGHT_BLUE = "#4DABF7"  # azul claro — classe 2 / segunda curva
ORANGE = "#E8590C"      # laranja — destaques


# ----------------------------------------------------------------------------- ativações
def tanh(u):
    return np.tanh(u)


def tanh_grad_from_output(h):
    """Derivada da tanh escrita em função da SAÍDA: d tanh(u)/du = 1 − tanh(u)² = 1 − h²."""
    return 1.0 - h ** 2


def sigmoid(u):
    """σ(u) = 1 / (1 + e^(−u)), calculada de forma estável para u muito negativo."""
    out = np.empty_like(u, dtype=float)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    eu = np.exp(u[~pos])                      # e^u, sem overflow quando u << 0
    out[~pos] = eu / (1.0 + eu)
    return out


def softmax(u):
    """softmax por linha. Subtrair o máximo não muda o resultado e evita overflow."""
    z = u - u.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------- perdas
EPS = 1e-12  # evita log(0)


def binary_cross_entropy(y_hat, y):
    """L = −(1/N) Σ [ y log ŷ + (1 − y) log(1 − ŷ) ],  y e ŷ com shape (N, 1)."""
    p = np.clip(y_hat, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def categorical_cross_entropy(y_hat, y_onehot):
    """L = −(1/N) Σ_n Σ_k y_nk log ŷ_nk,  com y em one-hot de shape (N, K)."""
    p = np.clip(y_hat, EPS, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))


# Cada tipo de saída é um PAR (ativação, perda). Os dois pares satisfazem dL/du = (ŷ − y)/N.
OUTPUTS = {
    "sigmoid": (sigmoid, binary_cross_entropy),
    "softmax": (softmax, categorical_cross_entropy),
}


# ------------------------------------------------------------------------------ utilidades
def one_hot(y, n_classes):
    """Rótulos inteiros (N,) -> matriz one-hot (N, K)."""
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def standardize(X_train, X_test):
    """z-score com média e desvio-padrão do TREINO (o teste não entra nas estatísticas)."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mu) / sd, (X_test - mu) / sd, (mu, sd)


# ------------------------------------------------------------------------------------ MLP
class MLP:
    """Perceptron multicamadas totalmente conectado.

    Parâmetros
    ----------
    layer_sizes : lista com o tamanho de cada camada, da entrada à saída. Exemplos:
        ``[2, 8, 1]`` (Ex. 2), ``[4, 16, 3]`` (Ex. 3), ``[4, 16, 16, 3]`` (Ex. 4).
        ``[2, 1]`` (nenhuma camada oculta) é exatamente a regressão logística.
    output : ``"sigmoid"`` (1 unidade, BCE) ou ``"softmax"`` (K unidades, CCE).
    rng : gerador ``np.random.default_rng`` usado na inicialização dos pesos.
    """

    def __init__(self, layer_sizes, output="sigmoid", rng=None):
        self.layer_sizes = list(layer_sizes)
        self.out_act, self.loss_fn = OUTPUTS[output]
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Inicialização de Xavier/Glorot (normal): Var(W) = 2 / (fan_in + fan_out),
        # adequada para tanh — mantém a variância das ativações parecida entre camadas.
        # Os vieses começam em zero.
        self.W, self.b = [], []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(self.rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    @property
    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ------------------------------------------------------------- passagem direta
    def forward(self, X):
        """Devolve a lista de ativações [h^(0)=X, h^(1), ..., h^(L)=ŷ].

        Camadas ocultas: h^(l) = tanh(h^(l−1) W^(l) + b^(l)).
        Última camada:   ŷ = σ(u^(L)) ou softmax(u^(L)).
        A lista é guardada porque a passagem reversa precisa de cada h^(l).
        """
        hs = [X]
        L = len(self.W)
        for l in range(L):
            u = hs[-1] @ self.W[l] + self.b[l]          # (N, fan_out)
            h = self.out_act(u) if l == L - 1 else tanh(u)
            hs.append(h)
        return hs

    # ------------------------------------------------------------- passagem reversa
    def backward(self, hs, Y):
        """Gradientes de L em relação a todos os W^(l) e b^(l), pela regra da cadeia.

        1. Saída: δ^(L) = dL/du^(L) = (ŷ − Y) / N   (vale para os dois pares de OUTPUTS).
        2. Para cada camada l, de trás para frente:
             dL/dW^(l) = h^(l−1)ᵀ δ^(l)        (soma sobre as N amostras)
             dL/db^(l) = Σ_n δ^(l)_n
             dL/dh^(l−1) = δ^(l) W^(l)ᵀ
             δ^(l−1) = dL/dh^(l−1) ⊙ (1 − (h^(l−1))²)     (derivada da tanh)
        """
        N = Y.shape[0]
        L = len(self.W)
        grads_W, grads_b = [None] * L, [None] * L
        delta = (hs[-1] - Y) / N                         # dL/du^(L)
        for l in range(L - 1, -1, -1):
            grads_W[l] = hs[l].T @ delta                 # (fan_in, fan_out)
            grads_b[l] = delta.sum(axis=0)               # (fan_out,)
            if l > 0:                                    # não há δ para a camada de entrada
                dh = delta @ self.W[l].T                 # dL/dh^(l)
                delta = dh * tanh_grad_from_output(hs[l])
        return grads_W, grads_b

    # ------------------------------------------------------------- atualização
    def update(self, grads_W, grads_b, lr):
        """Descida do gradiente: θ ← θ − η ∂L/∂θ, para todos os parâmetros."""
        for l in range(len(self.W)):
            self.W[l] -= lr * grads_W[l]
            self.b[l] -= lr * grads_b[l]

    # ------------------------------------------------------------- treino e predição
    def loss(self, X, Y):
        return self.loss_fn(self.forward(X)[-1], Y)

    def fit(self, X, Y, lr=0.1, epochs=1000):
        """Descida do gradiente em lote completo (full batch): uma época = uma passada
        direta + uma reversa + uma atualização com TODAS as amostras de treino.
        Guarda a perda de treino de cada época em ``self.loss_history``."""
        self.loss_history = []
        for _ in range(epochs):
            hs = self.forward(X)
            self.loss_history.append(self.loss_fn(hs[-1], Y))   # perda ANTES da atualização
            gW, gb = self.backward(hs, Y)
            self.update(gW, gb, lr)
        self.final_loss = self.loss(X, Y)                       # perda após a última época
        return self

    def predict_proba(self, X):
        return self.forward(X)[-1]

    def predict(self, X):
        """Classe prevista: limiar 0,5 (sigmoide) ou argmax (softmax)."""
        P = self.predict_proba(X)
        if P.shape[1] == 1:
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ------------------------------------------------------------------- checagem do gradiente
def gradient_check(model, X, Y, h=1e-6):
    """Compara os gradientes analíticos de ``backward`` com diferenças finitas centrais,
    [L(θ + h) − L(θ − h)] / 2h, parâmetro a parâmetro. Devolve o maior erro relativo."""
    gW, gb = model.backward(model.forward(X), Y)
    worst = 0.0
    for params, grads in ((model.W, gW), (model.b, gb)):
        for P, G in zip(params, grads):
            it = np.nditer(P, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                old = P[i]
                P[i] = old + h
                lp = model.loss(X, Y)
                P[i] = old - h
                lm = model.loss(X, Y)
                P[i] = old
                num = (lp - lm) / (2 * h)
                rel = abs(num - G[i]) / max(1e-8, abs(num) + abs(G[i]))
                worst = max(worst, rel)
    return worst
