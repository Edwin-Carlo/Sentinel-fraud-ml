"""
Contratos de datos (schemas) para la API de detección de fraude.

Los campos coinciden EXACTAMENTE con las 12 features que espera el modelo
entrenado (ver app_monitoring/model_robusto.pkl).
"""

from pydantic import BaseModel, Field


class TransaccionInput(BaseModel):
    """Datos de entrada de una transacción a evaluar."""

    # --- Variables base ---
    step: int = Field(..., description="Paso temporal (día simulado)")
    amount: float = Field(..., description="Monto de la transacción")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Día de la semana (0-6)")

    # --- Variables derivadas del modelo ---
    is_merchant_dest: int = Field(..., ge=0, le=1, description="1 si el destino es comercio")
    is_round_amount: int = Field(..., ge=0, le=1, description="1 si el monto es múltiplo de 1000")
    amount_log: float = Field(..., description="log1p(amount)")
    dest_frequency: int = Field(..., ge=0, description="Frecuencia de la cuenta destino")
    orig_frequency: int = Field(..., ge=0, description="Frecuencia de la cuenta origen")
    is_night_transaction: int = Field(..., ge=0, le=1, description="1 si es de madrugada (0-6h)")
    dest_is_merchant_and_transfer: int = Field(..., ge=0, le=1, description="Interacción comercio/transferencia")

    # --- One-Hot del tipo de transacción ---
    type_TRANSFER: int = Field(0, ge=0, le=1, description="1 si es TRANSFER")


class PrediccionOutput(BaseModel):
    """Datos de salida de la predicción."""

    transaccion_riesgo: bool
    probabilidad_fraude: float
    nivel_riesgo: str
    accion_sugerida: str