# Diccionario de Datos — PaySim Mobile Money Transactions

**Proyecto**: Sentinel-fraud-ml
**Dataset**: PaySim Synthetic Mobile Money Transactions
**Fuente**: [Kaggle — PaySim Synthetic Financial Datasets](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv)
**Registros**: 6.362.620 transacciones
**Columnas**: 11
**Variable objetivo**: `isFraud`
**Fecha de análisis**: Octubre 2026
**Autor**: Edwin Carlo

---

## Descripción general

PaySim es un dataset **sintético** generado por un simulador basado en datos reales agregados de un servicio de dinero móvil en África. Incluye un mes de transacciones financieras simuladas (744 horas ≈ 31 días) con comportamientos maliciosos inyectados para evaluar métodos de detección de fraude.

Es un dataset **tabular clásico**, apto para técnicas de machine learning supervisado de clasificación binaria.

---

## Diccionario de columnas

| # | Columna | Tipo | Descripción | Rango / Valores | Nulos |
|---|---------|------|-------------|-----------------|-------|
| 1 | `step` | int64 | Unidad de tiempo en horas. 1 step = 1 hora. 744 steps = 31 días | 1 – 743 | 0 |
| 2 | `type` | object | Tipo de transacción | CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER | 0 |
| 3 | `amount` | float64 | Monto de la transacción en moneda local | 0,00 – 92.445.516,64 | 0 |
| 4 | `nameOrig` | object | ID único del cliente origen | Cadena alfanumérica (ej. C1231006815) | 0 |
| 5 | `oldbalanceOrg` | float64 | Saldo del origen **antes** de la transacción | 0,00 – 59.585.040,37 | 0 |
| 6 | `newbalanceOrig` | float64 | Saldo del origen **después** de la transacción | 0,00 – 59.585.040,37 | 0 |
| 7 | `nameDest` | object | ID único del cliente destino | Cadena con prefijo M (merchant) o C (customer) | 0 |
| 8 | `oldbalanceDest` | float64 | Saldo del destino **antes** de la transacción | 0,00 – 356.015.889,35 | 0 |
| 9 | `newbalanceDest` | float64 | Saldo del destino **después** de la transacción | 0,00 – 356.179.278,92 | 0 |
| 10 | `isFraud` | int64 | **Variable objetivo**. Indica si la transacción es fraudulenta | 0 = No fraude, 1 = Fraude | 0 |
| 11 | `isFlaggedFraud` | int64 | Regla de negocio: marca transacciones > 200.000 | 0 = No marcado, 1 = Marcado | 0 |

---

## Tipos de transacción (`type`)

Análisis de la distribución de `type` y su relación con el fraude:

| Valor | Descripción | Frecuencia | Porcentaje | Fraudes | Tasa de fraude |
|-------|-------------|------------|-----------|---------|----------------|
| `CASH_OUT` | Retiro de efectivo | 2.237.500 | 35,17% | 4.116 | 0,1840% |
| `PAYMENT` | Pago a comercio | 2.151.495 | 33,81% | 0 | 0,0000% |
| `CASH_IN` | Depósito de efectivo | 1.399.284 | 21,99% | 0 | 0,0000% |
| `TRANSFER` | Transferencia entre cuentas | 532.909 | 8,38% | 4.097 | 0,7688% |
| `DEBIT` | Débito directo | 41.432 | 0,65% | 0 | 0,0000% |

**Observación clave**: el fraude **solo ocurre** en `TRANSFER` y `CASH_OUT`, coherente con el patrón de "transferir y retirar" del defraudador. Los tres tipos restantes (PAYMENT, CASH_IN, DEBIT) tienen tasa de fraude = 0%.

---

## Distribución de la variable objetivo

| Valor | Etiqueta | Frecuencia | Porcentaje |
|-------|----------|------------|-----------|
| 0 | No fraude | 6.354.407 | 99,8709% |
| 1 | Fraude | 8.213 | 0,1291% |

**Ratio de desbalance**: 1 fraude por cada ~774 transacciones legítimas.

**Implicación**: `Accuracy` es una métrica engañosa (un modelo trivial "todo no fraude" tendría 99,87% de accuracy). Se usarán:
- **F1-score** (principal)
- **PR-AUC** (mejor que ROC-AUC en desbalances extremos)
- **Recall** (crítico: no dejar pasar fraudes)
- **Precision** (controlar falsos positivos)

---

## Análisis de la regla de negocio `isFlaggedFraud`

| Métrica | Valor |
|---------|-------|
| Transacciones marcadas por regla | 16 |
| Fraudes reales | 8.213 |
| Cobertura de la regla | 0,1948% |
| Precisión de la regla (de los marcados, cuántos son fraude) | 100,00% |

**Interpretación**: la regla es **extremadamente conservadora**. Tiene precisión perfecta pero **recall casi nulo** (99,8% de fraudes pasan desapercibidos). Esto justifica plenamente la construcción de un modelo de ML.

---

## Distribución de montos (`amount`) por clase

| Estadística | No fraude | Fraude |
|-------------|-----------|--------|
| count | 6.354.407 | 8.213 |
| mean | 178.197,04 | **1.467.967,30** |
| std | 596.236,98 | 2.404.252,95 |
| min | 0,01 | 0,00 |
| 25% | 13.368,40 | 127.091,33 |
| 50% (mediana) | 74.684,72 | **441.423,44** |
| 75% | 208.364,76 | **1.517.771,48** |
| max | 92.445.516,64 | 10.000.000,00 |

**Observación**: los fraudes tienen una mediana **~6x mayor** que las transacciones legítimas. `amount` es una feature relevante.

---

## Análisis de Data Leakage

Se analizaron los errores de balance (`error_balance_orig` y `error_balance_dest`) en transacciones `TRANSFER` y `CASH_OUT`:

| Métrica | No fraude | Fraude |
|---------|-----------|--------|
| `error_balance_orig` (mediana) | 144.200,83 | **0,00** |
| `error_balance_dest` (mediana) | 0,00 | **9.511,69** |
| Ambos errores = 0 | 105.665 / 2.762.196 (3,83%) | 2.874 / 8.213 (**34,99%**) |

**Interpretación**:

- Los fraudes tienen **`error_balance_orig` = 0** en la mediana: el saldo del origen se agota completamente con el monto transferido (patrón de "vaciar la cuenta").
- Los no fraudes tienen **`error_balance_dest` = 0** en la mediana: la cuenta destino suele recibir el monto íntegro (sin comisión).
- El **35%** de los fraudes tienen ambos errores = 0 (vs. 3,83% de no fraudes): hay una señal fuerte, pero no perfecta.

**Riesgo**: un modelo entrenado con estas features aprendería atajos que **no generalizan** a patrones de fraude reales fuera de la simulación.

**Decisión crítica**: **NO crear features derivadas de errores de balance**. Documentado como decisión ética de feature engineering.

---

## Notas sobre calidad de datos

- ✅ **Sin valores nulos** en ninguna columna.
- ✅ **Sin duplicados exactos** de filas.
- ✅ **Sin valores fuera de rango** evidentes.
- ⚠️ **Riesgo de data leakage** identificado y neutralizado (ver sección anterior).

---

## Decisiones derivadas para el preprocesamiento

1. **Filtrar** a solo `TRANSFER` y `CASH_OUT` (donde ocurre el 100% del fraude). Reduce de 6,36M a 2,77M filas.
2. **Eliminar** `nameOrig`, `nameDest`, `isFlaggedFraud` (IDs y regla redundante).
3. **NO crear** features de error de balance (data leakage).
4. **Feature engineering ético**:
   - `hour_of_day` = `step % 24`
   - `day_of_week` = `(step // 24) % 7`
   - `amount_to_balance_ratio` = `amount / (oldbalanceOrg + 1)`
   - `is_merchant_dest` = `nameDest` empieza con "M"
5. **Balanceo**: `class_weight="balanced"` o SMOTE en train.
6. **Split estratificado** 70/15/15.

---

## Referencias

- [PaySim: A financial mobile money simulator for fraud detection](https://www.researchgate.net/publication/309723672_PaySim_A_financial_mobile_money_simulator_for_fraud_detection)
- [Dataset en Kaggle](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv)