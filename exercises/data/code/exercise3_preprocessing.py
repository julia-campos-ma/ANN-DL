"""Exercise 3 — Preparing Real-World Data for a Neural Network.

Carrega o dataset Spaceship Titanic (``data/train.csv``), descreve o dataset
(item A), faz o split treino/teste estratificado ANTES de qualquer estatística
(item B), pré-processa (imputação, one-hot, TotalSpend, log1p, escala — item
C) e salva a Figura 6 (antes/depois do log em FoodCourt).

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise3_preprocessing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = BASE_DIR / "figures"
DATA_PATH = BASE_DIR / "data" / "train.csv"
RNG = np.random.default_rng(42)

NUMERICAL_COLS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def describe_dataset(df: pd.DataFrame) -> None:
    """Item A — balanço do target, tipos de feature, missing e estatísticas de gasto."""
    target_balance = df["Transported"].value_counts(normalize=True)
    print("Distribuição de Transported (fração):")
    print(target_balance)

    missing = df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False)
    print("\nMissing values por coluna:")
    print(missing)

    spend_stats = df[SPEND_COLS].agg(["mean", "median", "max"]).T
    print("\nEstatísticas das colunas de gasto:")
    print(spend_stats)


def split_data(df: pd.DataFrame):
    """Item B — split ANTES de qualquer imputação/escala, seed derivada do RNG do relatório."""
    X_full = df.drop(columns=["Transported"])
    y_full = df["Transported"].astype(int)

    # train_test_split não aceita um objeto Generator como random_state — tira-se
    # um inteiro do MESMO rng do relatório em vez de recriar a seed 42 do zero.
    split_seed = int(RNG.integers(0, 1_000_000))

    return train_test_split(X_full, y_full, test_size=0.2, stratify=y_full,
                             random_state=split_seed)


def preprocess(X_train, X_test):
    """Item C — imputação, one-hot, TotalSpend, log1p, escala. Fit só no treino."""
    num_imputer = SimpleImputer(strategy="median")
    X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[NUMERICAL_COLS]),
                                columns=NUMERICAL_COLS, index=X_train.index)
    X_test_num = pd.DataFrame(num_imputer.transform(X_test[NUMERICAL_COLS]),
                               columns=NUMERICAL_COLS, index=X_test.index)

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[CATEGORICAL_COLS]),
                                columns=CATEGORICAL_COLS, index=X_train.index)
    X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[CATEGORICAL_COLS]),
                               columns=CATEGORICAL_COLS, index=X_test.index)

    # handle_unknown="ignore": categoria vista só no teste vira zero em todas
    # as colunas one-hot daquela feature, em vez de quebrar o .transform().
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_enc = pd.DataFrame(ohe.fit_transform(X_train_cat),
                                    columns=ohe.get_feature_names_out(CATEGORICAL_COLS),
                                    index=X_train.index)
    X_test_cat_enc = pd.DataFrame(ohe.transform(X_test_cat),
                                   columns=ohe.get_feature_names_out(CATEGORICAL_COLS),
                                   index=X_test.index)

    X_train_num["TotalSpend"] = X_train_num[SPEND_COLS].sum(axis=1)
    X_test_num["TotalSpend"] = X_test_num[SPEND_COLS].sum(axis=1)

    log_cols = SPEND_COLS + ["TotalSpend"]
    foodcourt_before = X_train_num["FoodCourt"].copy()  # (1)!
    for col in log_cols:
        X_train_num[col] = np.log1p(X_train_num[col])
        X_test_num[col] = np.log1p(X_test_num[col])

    # Standardization: o log1p acima já domou a cauda longa, então a maior parte
    # dos valores padronizados cai perto da região não saturada de tanh.
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num),
                                       columns=X_train_num.columns, index=X_train_num.index)
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num),
                                      columns=X_test_num.columns, index=X_test_num.index)

    X_train_final = pd.concat([X_train_num_scaled, X_train_cat_enc], axis=1)
    X_test_final = pd.concat([X_test_num_scaled, X_test_cat_enc], axis=1)

    return X_train_final, X_test_final, foodcourt_before, X_train_num_scaled["FoodCourt"]


def plot_figure6(foodcourt_before, foodcourt_after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(foodcourt_before, bins=40, color="tab:blue")
    axes[0].set_title("FoodCourt — antes do log(1+x)")
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("Frequência")

    axes[1].hist(foodcourt_after, bins=40, color="tab:orange")
    axes[1].set_title("FoodCourt — depois do log(1+x) e padronização")
    axes[1].set_xlabel("FoodCourt (padronizado)")
    axes[1].set_ylabel("Frequência")

    fig.suptitle("Figura 6 — Efeito do pré-processamento em FoodCourt")
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape}")
    describe_dataset(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTreino: {X_train.shape}, Teste: {X_test.shape}")

    X_train_final, X_test_final, fc_before, fc_after = preprocess(X_train, X_test)

    print(f"\nNaNs no treino: {X_train_final.isna().sum().sum()}")
    print(f"NaNs no teste:  {X_test_final.isna().sum().sum()}")
    print(f"Shape final treino: {X_train_final.shape}")
    print(f"Shape final teste:  {X_test_final.shape}")
    print(f"Faixa treino: [{X_train_final.values.min():.3f}, {X_train_final.values.max():.3f}]")
    print(f"Faixa teste:  [{X_test_final.values.min():.3f}, {X_test_final.values.max():.3f}]")

    fig6 = plot_figure6(fc_before, fc_after)
    fig6.savefig(FIGURES / "fig06-foodcourt-before-after.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)  # (2)!


if __name__ == "__main__":
    main()
