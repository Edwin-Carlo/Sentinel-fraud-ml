"""
Módulo de preprocesamiento para el proyecto Sentinel-fraud-ml.

Este módulo contiene funciones puras y reutilizables para transformar el
dataset crudo de PaySim en features listas para el modelado supervisado.

Diseño:
    - Cada función tiene una única responsabilidad.
    - Todas las funciones son puras (no modifican el DataFrame original).
    - Los docstrings siguen el estilo Google.
    - Los tipos están anotados para permitir mypy/ruff.

⚠️ CORRECCIÓN DE DATA LEAKAGE:

    Se detectó que las columnas de balance (`oldbalanceOrg`, `newbalanceOrig`,
    `oldbalanceDest`, `newbalanceDest`) y la feature derivada
    `amount_to_balance_ratio` causaban data leakage residual en el modelo:
    el Random Forest alcanzaba F1=0.997 dependiendo solo de 2 features.

    Estas columnas fueron ELIMINADAS. Ver `docs/correccion_leakage.md`.

Uso típico:
    >>> from src.preprocessing import (
    ...     load_data, filter_transactions, drop_unused_columns,
    ...     engineer_features, split_features_target, split_train_val_test,
    ... )
    >>> df = load_data("data/raw/dataset.csv")
    >>> df = filter_transactions(df)
    >>> df = engineer_features(df)     # antes de eliminar nameDest
    >>> df = drop_unused_columns(df)
    >>> X, y = split_features_target(df)
    >>> splits = split_train_val_test(X, y)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Constantes
# ============================================================

TARGET_COL: str = "isFraud"
RANDOM_SEED: int = 42

# Tipos de transacción donde ocurre el fraude (ver EDA)
VALID_TYPES: List[str] = ["TRANSFER", "CASH_OUT"]

# Columnas a eliminar del dataset tras el EDA
#
# Justificación de cada eliminación:
#   - nameOrig, nameDest       : IDs únicos no generalizables
#   - isFlaggedFraud           : regla de negocio redundante con la etiqueta
#   - oldbalanceOrg            : LEAKAGE — permite reconstruir ratio monto/saldo
#   - newbalanceOrig           : LEAKAGE — revela si la cuenta quedó en 0
#   - oldbalanceDest           : LEAKAGE — permite reconstruir ratio destino
#   - newbalanceDest           : LEAKAGE — revela si el receptor recibió todo
COLUMNS_TO_DROP: List[str] = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Tipos de datos que se esperan en cada columna categórica
CATEGORICAL_COLS: List[str] = ["type"]


# ============================================================
# Carga y filtrado
# ============================================================


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Carga el dataset crudo de PaySim desde un CSV.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos crudos.

    Raises:
        FileNotFoundError: si la ruta no existe.
        ValueError: si el DataFrame resultante está vacío.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El archivo {path} está vacío.")

    return df


def filter_transactions(
    df: pd.DataFrame,
    valid_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filtra transacciones a tipos donde ocurre el fraude.

    Según el EDA, el fraude solo ocurre en TRANSFER y CASH_OUT.
    Filtrar elimina el ruido de tipos sin fraude (PAYMENT, CASH_IN, DEBIT)
    y reduce el tamaño del dataset en ~56%.

    Args:
        df: DataFrame con los datos crudos.
        valid_types: Lista de tipos a mantener. Si es None,
            usa VALID_TYPES por defecto.

    Returns:
        DataFrame filtrado (copia).

    Raises:
        KeyError: si la columna 'type' no existe.
    """
    if "type" not in df.columns:
        raise KeyError("La columna 'type' no existe en el DataFrame.")

    valid_types = valid_types or VALID_TYPES
    return df[df["type"].isin(valid_types)].copy()


# ============================================================
# Feature engineering
# ============================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas éticas (sin leakage).

    Features creadas:
        - hour_of_day        : hora del día (0-23), calculada como step % 24.
        - day_of_week        : día de la semana (0-6), calculado como
                               (step // 24) % 7.
        - is_merchant_dest   : 1 si nameDest empieza con "M", 0 si no.

    ⚠️ NO se crea `amount_to_balance_ratio` (data leakage).

    IMPORTANTE: este paso se ejecuta ANTES de `drop_unused_columns`,
    porque necesita `nameDest` para calcular `is_merchant_dest`.

    Args:
        df: DataFrame de entrada.

    Returns:
        DataFrame con las nuevas features (copia).
    """
    df = df.copy()

    # --- Features temporales ---
    if "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        df["day_of_week"] = (df["step"] // 24) % 7

    # --- Flag de comercio en destino ---
    if "nameDest" in df.columns:
        df["is_merchant_dest"] = df["nameDest"].str.startswith("M").astype(int)

    return df


def drop_unused_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Elimina columnas no informativas y columnas con data leakage.

    Por defecto elimina las columnas de COLUMNS_TO_DROP:
        - IDs únicos (nameOrig, nameDest).
        - Regla de negocio redundante (isFlaggedFraud).
        - Columnas de balance con data leakage residual
          (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest).

    Si alguna columna no existe, se ignora silenciosamente.

    Args:
        df: DataFrame de entrada.
        columns: Lista de columnas a eliminar. Si es None,
            usa COLUMNS_TO_DROP por defecto.

    Returns:
        DataFrame sin las columnas indicadas (copia).
    """
    columns = columns or COLUMNS_TO_DROP
    cols_to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=cols_to_drop).copy()


def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features avanzadas sin leakage para mejorar la discriminación.

    Features creadas:
        - is_round_amount            : 1 si amount es múltiplo de 1000.
        - amount_log                 : log1p(amount) para reducir sesgo.
        - dest_frequency             : frecuencia de nameDest en el dataset.
        - orig_frequency             : frecuencia de nameOrig en el dataset.
        - is_night_transaction       : 1 si hour_of_day ∈ [0, 6].
        - dest_is_merchant_and_transfer : interacción binaria de type × is_merchant_dest.

    ⚠️ NO usa ninguna feature de balance (evita leakage).

    IMPORTANTE: debe ejecutarse ANTES de drop_unused_columns, porque
    necesita nameDest y nameOrig para calcular frecuencias.

    Args:
        df: DataFrame con las features básicas ya creadas.

    Returns:
        DataFrame con las features avanzadas añadidas.
    """
    df = df.copy()

    # --- 1. Monto redondo ---
    if "amount" in df.columns:
        df["is_round_amount"] = (df["amount"] % 1000 == 0).astype(int)
        df["amount_log"] = np.log1p(df["amount"])

    # --- 2. Frecuencias de IDs ---
    if "nameDest" in df.columns:
        df["dest_frequency"] = df["nameDest"].map(df["nameDest"].value_counts())
    if "nameOrig" in df.columns:
        df["orig_frequency"] = df["nameOrig"].map(df["nameOrig"].value_counts())

    # --- 3. Transacción nocturna ---
    if "hour_of_day" in df.columns:
        df["is_night_transaction"] = df["hour_of_day"].between(0, 6).astype(int)

    # --- 4. Interacción type × merchant ---
    if "type" in df.columns and "is_merchant_dest" in df.columns:
        df["dest_is_merchant_and_transfer"] = (
            (df["type"] == "TRANSFER") & (df["is_merchant_dest"] == 1)
        ).astype(int)

    return df


# ============================================================
# Split de features y target
# ============================================================


def split_features_target(
    df: pd.DataFrame,
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en features (X) y variable objetivo (y).

    Args:
        df: DataFrame completo.
        target: Nombre de la columna objetivo.

    Returns:
        Tupla (X, y).

    Raises:
        KeyError: si la columna objetivo no existe.
    """
    if target not in df.columns:
        raise KeyError(f"La columna objetivo '{target}' no existe.")

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Divide X e y en train, validación y test de forma estratificada.

    La estratificación mantiene la proporción de fraude en cada split,
    lo cual es crítico en datasets con desbalance extremo.

    Args:
        X: DataFrame de features.
        y: Serie con la variable objetivo.
        train_size: Proporción para train (default 0.70).
        val_size: Proporción para validación (default 0.15).
        test_size: Proporción para test (default 0.15).
        random_state: Semilla para reproducibilidad.

    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test).

    Raises:
        ValueError: si las proporciones no suman 1.0.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"Las proporciones deben sumar 1.0. "
            f"Recibido: {train_size + val_size + test_size:.4f}"
        )

    # Primer split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=random_state,
    )

    # Segundo split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
