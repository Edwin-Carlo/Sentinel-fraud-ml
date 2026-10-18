"""
API de detección de fraude en tiempo real (Sentinel Fraud ML).

Carga el modelo CalibratedClassifierCV entrenado y expone un endpoint /predict
que recibe datos de una transacción y devuelve la probabilidad de fraude
con el nivel de riesgo y la acción recomendada.
"""

import joblib
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from api.schemas import TransaccionInput, PrediccionOutput

# ============================================================
# Inicialización
# ============================================================

app = FastAPI(
    title="Sentinel Fraud API",
    description="API para detección de fraude en tiempo real (Modelo Calibrado).",
    version="1.0.0",
)

# Variables globales
MODELO = None
UMBRAL_FINANCIERO = 0.12  # Umbral óptimo de negocio (definido en el Paso 4)


@app.on_event("startup")
def cargar_modelo():
    """Carga el modelo entrenado al arrancar la API."""
    global MODELO
    try:
        ruta_modelo = Path(__file__).parent.parent / "app_monitoring" / "model_robusto.pkl"
        if not ruta_modelo.exists():
            print(f"❌ No se encontró el modelo en {ruta_modelo}")
            return
        MODELO = joblib.load(ruta_modelo)
        print(f"✅ Modelo cargado desde: {ruta_modelo}")
        print(f"   Tipo: {type(MODELO).__name__}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")


# ============================================================
# Endpoints
# ============================================================


@app.get("/health")
def health_check():
    """Endpoint de salud para monitoreo."""
    return {
        "status": "ok",
        "modelo_cargado": MODELO is not None,
    }


@app.post("/predict", response_model=PrediccionOutput)
def predecir_fraude(transaccion: TransaccionInput):
    """Predice si una transacción es fraudulenta.

    Args:
        transaccion: Datos de la transacción a evaluar.

    Returns:
        Predicción con probabilidad, nivel de riesgo y acción sugerida.

    Raises:
        HTTPException: si el modelo no está cargado.
    """
    if MODELO is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo no está cargado en el servidor.",
        )

    # Convertir el JSON a DataFrame
    datos = pd.DataFrame([transaccion.model_dump()])

    # Asegurar que las columnas coincidan con las del modelo
    if hasattr(MODELO, "feature_names_in_"):
        columnas = MODELO.feature_names_in_
    elif hasattr(MODELO, "estimator") and hasattr(MODELO.estimator, "feature_names_in_"):
        columnas = MODELO.estimator.feature_names_in_
    else:
        columnas = datos.columns

    datos = datos.reindex(columns=columnas, fill_value=0)

    # Predecir
    probabilidad = float(MODELO.predict_proba(datos)[0][1])
    es_fraude = probabilidad >= UMBRAL_FINANCIERO

    # Reglas de negocio
    if probabilidad >= 0.80:
        nivel = "CRÍTICO"
        accion = "BLOQUEO INMEDIATO"
    elif probabilidad >= UMBRAL_FINANCIERO:
        nivel = "ALTO"
        accion = "REVISIÓN MANUAL URGENTE"
    elif probabilidad >= 0.05:
        nivel = "MEDIO"
        accion = "MONITOREAR"
    else:
        nivel = "BAJO"
        accion = "APROBAR"

    return PrediccionOutput(
        transaccion_riesgo=es_fraude,
        probabilidad_fraude=round(probabilidad, 4),
        nivel_riesgo=nivel,
        accion_sugerida=accion,
    )