"""
Tests unitarios para el módulo src/preprocessing.py.

Ejecutar con:
    uv run pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    drop_unused_columns,
    engineer_features,
    filter_transactions,
    split_features_target,
    split_train_val_test,
    COLUMNS_TO_DROP,
    TARGET_COL,
    VALID_TYPES,
)


# ============================================================
# Fixture: DataFrame de prueba
# ============================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame mínimo que simula el dataset de PaySim.

    Contiene 6 filas con 3 TRANSFER/CASH_OUT y 3 PAYMENT/CASH_IN/DEBIT.
    Los fraudes (isFraud=1) están en TRANSFER y CASH_OUT.
    """
    return pd.DataFrame({
        "step":           [1, 2, 3, 4, 5, 6],
        "type":           ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT", "TRANSFER"],
        "amount":         [100.0, 200.0, 50.0, 30.0, 10.0, 500.0],
        "nameOrig":       ["C1", "C2", "C3", "C4", "C5", "C6"],
        "oldbalanceOrg":  [1000.0, 500.0, 200.0, 100.0, 50.0, 2000.0],
        "newbalanceOrig": [900.0, 300.0, 150.0, 70.0, 40.0, 1500.0],
        "nameDest":       ["M1", "C10", "M2", "C20", "M3", "C30"],
        "oldbalanceDest": [0.0, 100.0, 0.0, 50.0, 0.0, 0.0],
        "newbalanceDest": [100.0, 300.0, 50.0, 80.0, 10.0, 500.0],
        "isFraud":        [1, 1, 0, 0, 0, 0],
        "isFlaggedFraud": [0, 0, 0, 0, 0, 0],
    })


# ============================================================
# Tests: filter_transactions
# ============================================================


def test_filter_transactions_keeps_only_valid_types(sample_df):
    """Debe mantener solo TRANSFER y CASH_OUT."""
    result = filter_transactions(sample_df)
    assert set(result["type"].unique()).issubset(set(VALID_TYPES))
    # En el fixture hay 2 TRANSFER + 1 CASH_OUT = 3 filas
    assert len(result) == 3


def test_filter_transactions_raises_if_no_type_column(sample_df):
    """Debe lanzar KeyError si falta la columna 'type'."""
    df_no_type = sample_df.drop(columns=["type"])
    with pytest.raises(KeyError):
        filter_transactions(df_no_type)


# ============================================================
# Tests: drop_unused_columns
# ============================================================


def test_drop_unused_columns_removes_expected(sample_df):
    """Debe eliminar nameOrig, nameDest, isFlaggedFraud."""
    result = drop_unused_columns(sample_df)
    for col in COLUMNS_TO_DROP:
        assert col not in result.columns


def test_drop_unused_columns_ignores_missing(sample_df):
    """No debe fallar si alguna columna no existe."""
    df_partial = sample_df.drop(columns=["nameOrig"])
    result = drop_unused_columns(df_partial)
    assert "nameDest" not in result.columns
    assert "isFlaggedFraud" not in result.columns


# ============================================================
# Tests: engineer_features
# ============================================================


def test_engineer_features_creates_expected_columns(sample_df):
    """Debe crear las 4 features derivadas esperadas."""
    result = engineer_features(sample_df)
    expected = {
        "hour_of_day",
        "day_of_week",
        "amount_to_balance_ratio",
        "is_merchant_dest",
    }
    assert expected.issubset(set(result.columns))


def test_engineer_features_hour_range(sample_df):
    """hour_of_day debe estar en el rango [0, 23]."""
    result = engineer_features(sample_df)
    assert result["hour_of_day"].between(0, 23).all()


def test_engineer_features_is_merchant_dest(sample_df):
    """is_merchant_dest debe ser 1 si nameDest empieza con 'M'."""
    result = engineer_features(sample_df)
    expected = sample_df["nameDest"].str.startswith("M").astype(int).values
    np.testing.assert_array_equal(
        result["is_merchant_dest"].values,
        expected,
    )


def test_engineer_features_no_leakage_columns(sample_df):
    """NO debe crear features de error de balance (data leakage)."""
    result = engineer_features(sample_df)
    forbidden = {
        "error_balance_orig",
        "error_balance_dest",
        "balance_change_orig",
    }
    assert not forbidden.intersection(set(result.columns))


# ============================================================
# Tests: split_features_target
# ============================================================


def test_split_features_target(sample_df):
    """Debe separar X e y correctamente."""
    X, y = split_features_target(sample_df)
    assert TARGET_COL not in X.columns
    assert y.name == TARGET_COL
    assert len(X) == len(y)


def test_split_features_target_raises_if_missing(sample_df):
    """Debe lanzar KeyError si falta la columna objetivo."""
    df_no_target = sample_df.drop(columns=[TARGET_COL])
    with pytest.raises(KeyError):
        split_features_target(df_no_target)


# ============================================================
# Tests: split_train_val_test
# ============================================================


def test_split_train_val_test_shapes(sample_df):
    """Debe generar tres splits con las proporciones correctas."""
    big_df = pd.concat([sample_df] * 20, ignore_index=True)
    X, y = split_features_target(big_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15,
    )

    total = len(X)
    assert len(X_train) + len(X_val) + len(X_test) == total


def test_split_train_val_test_stratified(sample_df):
    """Debe mantener la proporción de clases (estratificación)."""
    big_df = pd.concat([sample_df] * 20, ignore_index=True)
    X, y = split_features_target(big_df)

    _, _, _, y_train, y_val, y_test = split_train_val_test(X, y)

    ratios = [y_train.mean(), y_val.mean(), y_test.mean()]
    assert max(ratios) - min(ratios) < 0.05


def test_split_train_val_test_bad_proportions(sample_df):
    """Debe lanzar ValueError si las proporciones no suman 1.0."""
    X, y = split_features_target(sample_df)
    with pytest.raises(ValueError):
        split_train_val_test(
            X, y,
            train_size=0.50,
            val_size=0.20,
            test_size=0.20,
        )