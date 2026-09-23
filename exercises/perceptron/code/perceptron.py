"""Perceptron de camada única, escrito à mão (só NumPy).

Este módulo é usado sem nenhuma alteração pelos Exercises 1 e 2 do enunciado:
a regra de atualização e o laço de treino são os mesmos; o que muda entre os
exercícios é apenas o dado. Ele contém:

    * ``generate_two_classes`` — geração das duas classes gaussianas 2D;
    * ``step``                 — ativação degrau;
    * ``decision_line``        — reta w·x + b = 0 para desenhar a fronteira;
    * ``Perceptron``           — predição, regra de atualização, laço de treino e
                                 rastreamento do "bolso" (pocket).
"""

import numpy as np

# Paleta usada em todas as figuras do exercício: só tons de rosa, azul e laranja.
NAVY = "#173F5F"        # azul escuro — classe 0
PINK = "#D6336C"        # rosa — classe 1
LIGHT_BLUE = "#4DABF7"  # azul claro — segunda linha/curva
ORANGE = "#E8590C"      # laranja — erros, fronteiras e destaques


def generate_two_classes(mean0, mean1, var, n_per_class, rng):
    """Gera duas classes 2D a partir de normais multivariadas de covariância ``var * I``.

    Retorna X com shape (2 * n_per_class, 2) e y com rótulos {0, 1}. As amostras da
    classe 0 vêm primeiro e as da classe 1 depois — a ordem é mantida no treino.
    """
    cov = np.array([[var, 0.0], [0.0, var]])  # covariância diagonal: eixos independentes
    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def step(z):
    """Degrau: 1 se z >= 0, senão 0 (funciona para escalar ou array)."""
    return (z >= 0).astype(int)


def decision_line(w, b, x1):
    """Pontos (x1, x2) da fronteira w·x + b = 0, isolando x2 = -(w1·x1 + b) / w2.
    -> usada só pra desenhar a fronteira"""
    return -(w[0] * x1 + b) / w[1]


class Perceptron:
    """Perceptron com ativação degrau e rótulos y ∈ {0, 1}.

    Parâmetros
    ----------
    lr : taxa de aprendizado η.
    max_epochs : número máximo de épocas (uma época = uma passada por todas as amostras).
    rng : gerador NumPy usado só na inicialização de w.
    init_std : desvio-padrão da inicialização de w (padrão 0.01, como no enunciado).
        Só é alterado para 0 na verificação algébrica do item D.3 (partida em w = 0).
    """

    def __init__(self, lr=0.01, max_epochs=100, rng=None, init_std=0.01):
        self.lr = lr
        self.max_epochs = max_epochs
        self.init_std = init_std
        self.rng = rng if rng is not None else np.random.default_rng(42)

    # ---------------------------------------------------------------- predição
    def predict(self, X, w=None, b=None):
        """ŷ = step(w·x + b). Por padrão usa os pesos atuais; w e b podem ser passados
        explicitamente para avaliar outros pesos (ex.: os do bolso)."""
        w = self.w if w is None else w
        b = self.b if b is None else b
        return step(X @ w + b)

    def accuracy(self, X, y, w=None, b=None):
        """Fração de amostras classificadas corretamente."""
        return float(np.mean(self.predict(X, w, b) == y))

    # ------------------------------------------------------------------ treino
    def fit(self, X, y):
        """Treina com atualização online (amostra a amostra), sempre na mesma ordem.

        Regra de atualização (erro ∈ {0, +1, -1}):
            w ← w + η (y − ŷ) x
            b ← b + η (y − ŷ)

        Parada: uma época inteira sem nenhuma atualização (convergência) ou
        ``max_epochs`` épocas.

        Guarda também o "bolso" (pocket): o melhor (w, b) já visto, medido pela
        acurácia no conjunto completo. Como (w, b) só muda quando há uma atualização,
        a acurácia é reavaliada logo após CADA atualização — e o bolso só é substituído
        se a acurácia for estritamente maior que a do bolso atual. Em dados separáveis
        o bolso acaba com a mesma acurácia dos pesos finais; ele só importa quando não
        há convergência.
        """
        # Inicialização não nula: w ~ N(0, 0.01²), b = 0
        self.w = self.rng.normal(0.0, self.init_std, size=X.shape[1])
        self.b = 0.0

        self.acc_history = []        # acurácia dos pesos ATUAIS ao fim de cada época
        self.best_acc_history = []   # melhor acurácia até a época (curva do bolso)
        self.updates_history = []    # nº de atualizações (erros) em cada época
        self.w_history = [self.w.copy()]  # w antes do treino e após cada época
        self.b_history = [self.b]

        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_acc = self.accuracy(X, y)   # começa pelo chute inicial
        self.pocket_epoch = 0                   # época em que o bolso foi salvo (0 = pesos iniciais)
        self.pocket_update = 0                  # nº da atualização em que o bolso foi salvo
        n_total_updates = 0
        self.converged = False
        self.epochs_run = 0
        self.total_updates = 0

        for epoch in range(1, self.max_epochs + 1):
            n_updates = 0
            for xi, yi in zip(X, y):
                y_hat = int(step(xi @ self.w + self.b))
                err = yi - y_hat            # 0 se acertou; ±1 se errou
                if err != 0:
                    self.w = self.w + self.lr * err * xi
                    self.b = self.b + self.lr * err
                    n_updates += 1
                    n_total_updates += 1

                    # Bolso: (w, b) acabou de mudar, então reavalia no dataset inteiro.
                    # ">" estrito: empates não substituem o bolso.
                    acc_now = self.accuracy(X, y)
                    if acc_now > self.pocket_acc:
                        self.pocket_w, self.pocket_b = self.w.copy(), self.b
                        self.pocket_acc = acc_now
                        self.pocket_epoch = epoch
                        self.pocket_update = n_total_updates

            acc = self.accuracy(X, y)
            self.acc_history.append(acc)
            self.updates_history.append(n_updates)
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)
            self.epochs_run = epoch

            self.best_acc_history.append(self.pocket_acc)  # curva do bolso ao fim da época

            if n_updates == 0:              # passada limpa: convergiu
                self.converged = True
                break
        self.total_updates = n_total_updates
        return self
