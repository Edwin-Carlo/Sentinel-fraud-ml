"""
Módulo de entrenamiento y optimización de hiperparámetros para el
proyecto Sentinel-fraud-ml.

Contiene:
    - Registro de modelos (Logistic Regression, Random Forest, Gradient Boosting).
    - Espacios de búsqueda de hiperparámetros por modelo.
    - Utilidades de evaluación y persistencia.

Uso típico:
    >>> from src.training import MODEL_REGISTRY, get_search_space
    >>> params = get_search_space("random_forest", trial)
    >>> model = MODEL_REGISTRY["random_forest"](**params)
"""

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# Constantes
# ============================================================

TARGET_COL = "isFraud"
RANDOM_SEED = 42

# ============================================================
# Registro de modelos
# ============================================================

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# Parámetros comunes por modelo
MODEL_DEFAULTS = {
    "logistic_regression": {
        "random_state": RANDOM_SEED,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
    },
    "random_forest": {
        "random_state": RANDOM_SEED,
        "n_jobs": 2,
        "class_weight": "balanced",
    },
    "gradient_boosting": {
        "random_state": RANDOM_SEED,
    },
}


# ============================================================
# Espacios de búsqueda por modelo
# ============================================================


def get_search_space(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Define el espacio de búsqueda de hiperparámetros por modelo.

    Args:
        model_name: Nombre del modelo (clave de MODEL_REGISTRY).
        trial: Objeto Trial de Optuna.

    Returns:
        Diccionario con hiperparámetros sugeridos.

    Raises:
        ValueError: si el modelo no está registrado.
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    raise ValueError(
        f"Modelo '{model_name}' no registrado. "
        f"Opciones válidas: {list(MODEL_REGISTRY.keys())}"
    )


# ============================================================
# Construcción del modelo
# ============================================================


def build_model(model_name: str, params: Dict[str, Any]):
    """Construye una instancia del modelo con los parámetros dados.

    Args:
        model_name: Nombre del modelo.
        params: Diccionario de hiperparámetros.

    Returns:
        Instancia del clasificador.

    Raises:
        ValueError: si el modelo no está registrado.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{model_name}' no registrado.")

    defaults = MODEL_DEFAULTS.get(model_name, {})
    merged = {**defaults, **params}
    return MODEL_REGISTRY[model_name](**merged)


# ============================================================
# Evaluación
# ============================================================


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Calcula métricas de evaluación para clasificación binaria.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas (0/1).
        y_proba: Probabilidades predichas.

    Returns:
        Diccionario con F1, Precision, Recall, ROC-AUC y PR-AUC.
    """
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


# ============================================================
# Persistencia
# ============================================================


def save_model(model, path: str | Path) -> Path:
    """Guarda un modelo entrenado en disco.

    Args:
        model: Modelo sklearn entrenado.
        path: Ruta donde guardar.

    Returns:
        Path absoluto del archivo guardado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path.resolve()


def load_model(path: str | Path):
    """Carga un modelo guardado.

    Args:
        path: Ruta al archivo.

    Returns:
        Modelo cargado.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    return joblib.load(path)



def get_sample_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    """Calcula pesos de muestra balanceados para clasificación desbalanceada.

    Asigna mayor peso a la clase minoritaria para que el modelo
    preste más atención a los fraudes durante el entrenamiento.

    Args:
        y: Etiquetas de la clase (0/1).

    Returns:
        Array de pesos, uno por muestra.

    Notas:
        - Se usa `compute_sample_weight("balanced", y)` de sklearn.
        - El peso de la clase minoritaria se multiplica por (n_mayoritaria / n_minoritaria).
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight(class_weight="balanced", y=y)