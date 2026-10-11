# Resumen del Análisis Exploratorio de Datos (EDA)

**Proyecto**: Sentinel-fraud-ml
**Notebook fuente**: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)
**Dataset**: PaySim Synthetic Mobile Money Transactions
**Autor**: Edwin Carlo
**Fecha**: Octubre 2026

---

## Objetivo

Documentar de forma ejecutiva los hallazgos del análisis exploratorio del dataset PaySim, las decisiones derivadas y los riesgos identificados para las etapas posteriores del proyecto.

---

## 1. Estructura del dataset

| Métrica | Valor |
|---------|-------|
| Filas | 6.362.620 |
| Columnas | 11 |
| Memoria | ~1,5 GB |
| Valores nulos | 0 |
| Duplicados | 0 |
| Tipos de datos | 5 int/float + 3 string + 3 float |

**Conclusión**: dataset limpio, no requiere imputación ni deduplicación.

---

## 2. Desbalance extremo de clases

| Clase | Frecuencia | Porcentaje |
|-------|------------|-----------|
| No fraude (0) | 6.354.407 | 99,8709% |
| Fraude (1) | 8.213 | 0,1291% |

**Conclusión**: usar **F1-score** y **PR-AUC**. Descartar `Accuracy`.

---

## 3. Fraude concentrado en TRANSFER y CASH_OUT

| Tipo | Total | Fraudes | Tasa |
|------|-------|---------|------|
| TRANSFER | 532.909 | 4.097 | 0,77% |
| CASH_OUT | 2.237.500 | 4.116 | 0,18% |
| CASH_IN | 1.399.284 | 0 | 0,00% |
| DEBIT | 41.432 | 0 | 0,00% |
| PAYMENT | 2.151.495 | 0 | 0,00% |

**Conclusión**: filtrar el dataset a `TRANSFER` + `CASH_OUT`. Reduce de 6,36M a **2,77M filas** (56% menos), sin perder ningún fraude.

---

## 4. Regla de negocio `isFlaggedFraud` inefectiva

| Métrica | Valor |
|---------|-------|
| Transacciones marcadas | 16 |
| Fraudes reales | 8.213 |
| Cobertura | **0,1948%** |
| Precisión de la regla | 100,00% |

**Conclusión**: la regla es demasiado conservadora. Su precisión perfecta no compensa su recall casi nulo. **Justifica plenamente un modelo de ML**.

---

## 5. Montos de fraude significativamente mayores

| Estadística | No fraude | Fraude | Ratio |
|-------------|-----------|--------|-------|
| Mediana | 74.684,72 | 441.423,44 | **5,9x** |
| Media | 178.197,04 | 1.467.967,30 | 8,2x |
| P75 | 208.364,76 | 1.517.771,48 | 7,3x |

**Conclusión**: `amount` es una feature relevante. Se creará `amount_to_balance_ratio` como derivada.

---

## 6. ⚠️ Data leakage detectado

Análisis en transacciones TRANSFER + CASH_OUT:

| Métrica | No fraude | Fraude |
|---------|-----------|--------|
| `error_balance_orig` (mediana) | 144.200,83 | **0,00** |
| `error_balance_dest` (mediana) | 0,00 | **9.511,69** |
| Ambos errores = 0 | 3,83% | **34,99%** |

**Interpretación**:

- Los fraudes tienen `error_balance_orig ≈ 0` (patrón de "vaciar la cuenta").
- Los no fraudes tienen `error_balance_dest ≈ 0` (transacción sin comisión).
- El 35% de fraudes tienen ambos errores = 0 (vs 3,83% de no fraudes).
- La señal es **fuerte pero no perfecta**. Un modelo que la use aprendería un atajo que no generaliza.

**Decisión**: NO crear features de error de balance. Documentado como decisión ética de feature engineering.

---

## 7. Decisiones para el preprocesamiento (Paso 3)

### Filtrado
- Mantener solo `TRANSFER` y `CASH_OUT`.
- Reducción: 6,36M → 2,77M filas.

### Eliminación de columnas
- `nameOrig`, `nameDest` — IDs únicos no generalizables.
- `isFlaggedFraud` — regla de negocio redundante con la etiqueta.

### Feature engineering ético
- `hour_of_day` = `step % 24`
- `day_of_week` = `(step // 24) % 7`
- `amount_to_balance_ratio` = `amount / (oldbalanceOrg + 1)`
- `is_merchant_dest` = `nameDest` empieza con "M"

### Features que NO se crearán (por leakage)
- `error_balance_orig`
- `error_balance_dest`
- `balance_change_orig`
- Cualquier combinación lineal de saldos y monto que aproxime errores.

### Estrategia de datos
- Split estratificado 70/15/15 (train/val/test).
- Balanceo: `class_weight="balanced"` o SMOTE **solo en train**.

### Métricas objetivo
- **F1-score** (principal)
- **PR-AUC**
- **Recall** (crítico en fraude)
- **Precision**

---

## 8. Riesgos identificados para monitoreo (Paso 6)

| Riesgo | Descripción | Mitigación |
|--------|-------------|-----------|
| **Data drift** | Cambios en distribución de `type` o `amount` | Evidently (PSI, KS) |
| **Concept drift** | Cambios en tácticas de fraude | Monitoreo de F1/Recall en el tiempo |
| **Model drift** | Cambios en distribución de predicciones | Monitoreo de media/varianza |
| **Umbral crítico** | Recall < 80% | Alerta y reentrenamiento |

---

## 9. Recursos generados por el EDA

### Figuras
- `reports/figures/eda_target_distribution.png` — Distribución del target
- `reports/figures/eda_type_distribution.png` — Distribución de tipos de transacción y tasa de fraude
- `reports/figures/eda_amount_distribution.png` — Distribución de montos por clase

### Notebook completo
- `notebooks/01_eda.ipynb`

---

## Conclusión

El dataset es de **alta calidad sintética** con **desbalance extremo** y un **riesgo de data leakage identificado y neutralizado**. Los hallazgos definen una estrategia sólida para el preprocesamiento (Paso 3) y el modelado (Paso 4), con foco en **F1-score**, **Recall** y **PR-AUC** como métricas primarias.

**Próximo paso**: Paso 3 — Construcción del pipeline de preprocesamiento en `src/preprocessing.py` con tests unitarios.