"""
Módulo de preprocesamiento para el proyecto Sentinel-fraud-ml.

Este módulo contiene funciones puras y reutilizables para transformar el
dataset crudo de PaySim en features listas para el modelado. Cada función
tiene una única responsabilidad y está documentada para permitir testing
y reutilización fuera del notebook.

Uso típico:
    >>> from src.preprocessing import (
    ...     load_data, filter_transactions, drop_unused_columns,
    ...     engineer_features, split_features_target, split_train_val_test,
    ... )
    >>> df = load_data("data/raw/dataset.csv")
    >>> df = filter_transactions(df)
    >>> df = drop_unused_columns(df)
    >>> df = engineer_features(df)
    >>> X, y = split_features_target(df)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Constantes
# ============================================================

TARGET_COL = "isFraud"
VALID_TYPES = ["TRANSFER", "CASH_OUT"]
RANDOM_SEED = 42
COLUMNS_TO_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]


# ============================================================
# Funciones
# ============================================================


def load_data(path: str | Path) -> pd.DataFrame:
    """Carga el dataset crudo de PaySim desde un CSV.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos crudos.

    Raises:
        FileNotFoundError: si la ruta no existe.
        ValueError: si el DataFrame está vacío.
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
    valid_types: list[str] | None = None,
) -> pd.DataFrame:
    """Filtra transacciones a tipos donde ocurre fraude.

    Según el EDA, el fraude solo ocurre en TRANSFER y CASH_OUT.

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


def drop_unused_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Elimina columnas no informativas o redundantes.

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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas a partir de las columnas originales.

    Features creadas:
        - hour_of_day: hora del día (0-23) derivada de step.
        - day_of_week: día de la semana (0-6) derivado de step.
        - amount_to_balance_ratio: monto dividido por saldo origen.
        - is_merchant_dest: 1 si nameDest empieza con "M", 0 si no.

    IMPORTANTE: NO se crean features de error de balance para evitar
    data leakage (ver docs/eda_resumen.md).

    Args:
        df: DataFrame de entrada.

    Returns:
        DataFrame con nuevas features (copia).
    """
    df = df.copy()

    if "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        df["day_of_week"] = (df["step"] // 24) % 7

    if "amount" in df.columns and "oldbalanceOrg" in df.columns:
        df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    if "nameDest" in df.columns:
        df["is_merchant_dest"] = df["nameDest"].str.startswith("M").astype(int)

    return df


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

    Args:
        X: DataFrame de features.
        y: Serie con la variable objetivo.
        train_size: Proporción para train.
        val_size: Proporción para validación.
        test_size: Proporción para test.
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

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test