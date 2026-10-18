import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset
import os

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA (ANCHO COMPLETO)
# ==============================================================================
st.set_page_config(page_title="Sentinel Fraud ML", page_icon="🛡️", layout="wide")

st.title("🛡️ Sentinel Fraud ML — Panel de Monitoreo")
st.markdown("""
Bienvenido al centro de control del modelo de detección de fraude.
Aquí puedes simular transacciones en tiempo real y monitorear la estabilidad de los datos.
""")

# ==============================================================================
# CARGA DE MODELO Y DATOS
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    ruta_modelo = Path(__file__).parent / "model_robusto.pkl"
    if not ruta_modelo.exists():
        st.error(f"❌ No se encontró el modelo en {ruta_modelo}")
        return None, None
    modelo = joblib.load(ruta_modelo)
    clasificador = modelo.estimator if hasattr(modelo, 'estimator') else modelo
    explainer = shap.TreeExplainer(clasificador)
    return modelo, explainer

@st.cache_data
def cargar_datos_referencia():
    ruta_train = Path(__file__).parent.parent / "data" / "processed" / "train.parquet"
    if ruta_train.exists():
        df = pd.read_parquet(ruta_train)
        return df.sample(n=min(10000, len(df)), random_state=42)
    return None

modelo, explainer = cargar_modelo()
df_referencia = cargar_datos_referencia()

# ==============================================================================
# PESTAÑAS
# ==============================================================================
tab1, tab2 = st.tabs(["🔮 Simulador en Vivo", "📊 Monitoreo y Análisis Masivo"])

# ==============================================================================
# PESTAÑA 1: SIMULADOR
# ==============================================================================
with tab1:
    st.header("Simulador de Transacciones")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Monto (Bs)", min_value=0.0, value=50000.0)
        hour = st.slider("Hora del día", 0, 23, 4)
        step = st.number_input("Paso temporal", min_value=1, value=650)
    with col2:
        type_op = st.selectbox("Tipo de operación", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
        dest_freq = st.number_input("Frecuencia destino", min_value=0, value=5)
        orig_freq = st.number_input("Frecuencia origen", min_value=0, value=50)
    with col3:
        oldbalanceOrg = st.number_input("Saldo previo origen", min_value=0.0, value=100000.0)
        newbalanceOrig = st.number_input("Saldo nuevo origen", min_value=0.0, value=50000.0)
        oldbalanceDest = st.number_input("Saldo previo destino", min_value=0.0, value=0.0)

    if st.button("Evaluar Transacción", type="primary"):
        if modelo is not None:
            amount_log = np.log1p(amount)
            is_round = 1 if amount % 1000 == 0 else 0
            is_night = 1 if hour <= 6 else 0
            is_merchant = 1 if type_op == "PAYMENT" else 0
            
            datos_crudos = {
                "step": step, "amount": amount, "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig, "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": oldbalanceDest + amount, "amount_log": amount_log,
                "hour_of_day": hour, "day_of_week": 1, "is_round_amount": is_round,
                "is_night_transaction": is_night, "dest_frequency": dest_freq,
                "orig_frequency": orig_freq,
                "dest_is_merchant_and_transfer": is_merchant if type_op == "TRANSFER" else 0,
                "is_merchant_dest": is_merchant,
                "type_TRANSFER": 1 if type_op == "TRANSFER" else 0,
                "type_CASH_OUT": 1 if type_op == "CASH_OUT" else 0,
                "type_PAYMENT": 1 if type_op == "PAYMENT" else 0,
                "type_DEBIT": 1 if type_op == "DEBIT" else 0
            }
            
            datos = pd.DataFrame([datos_crudos])
            
            if hasattr(modelo, 'feature_names_in_'):
                columnas_modelo = modelo.feature_names_in_
            elif hasattr(modelo.estimator, 'feature_names_in_'):
                columnas_modelo = modelo.estimator.feature_names_in_
            else:
                columnas_modelo = datos.columns
                
            datos = datos.reindex(columns=columnas_modelo, fill_value=0)

            probabilidad = modelo.predict_proba(datos)[0][1]
            umbral = 0.12
            
            st.divider()
            if probabilidad >= umbral:
                st.error(f"🚨 **ALERTA DE FRAUDE** (Probabilidad: {probabilidad:.2%})")
            else:
                st.success(f"✅ **TRANSACCIÓN LEGÍTIMA** (Probabilidad: {probabilidad:.2%})")

            st.subheader("🔍 Explicación SHAP")
            shap_values = explainer.shap_values(datos)[0]
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.waterfall_plot(
                shap.Explanation(values=shap_values, base_values=explainer.expected_value, 
                                 data=datos.iloc[0], feature_names=datos.columns),
                max_display=6, show=False
            )
            st.pyplot(fig)

# ==============================================================================
# PESTAÑA 2: MONITOREO Y ANÁLISIS MASIVO
# ==============================================================================
with tab2:
    st.header("Monitoreo de Data Drift y Análisis de Lote")
    
    archivo_subido = st.file_uploader("Sube un archivo CSV de producción", type=["csv"])
    
    if archivo_subido is not None and df_referencia is not None:
        df_produccion = pd.read_csv(archivo_subido)
        
        # ======================================================================
        # 1. RESUMEN DE LA BASE
        # ======================================================================
        st.subheader("1. Resumen de la Base de Producción")
        
        col_a, col_b = st.columns(2)
        col_a.metric("Total Transacciones", f"{len(df_produccion):,}")
        col_b.metric("Columnas Detectadas", f"{len(df_produccion.columns)}")
            
        with st.expander("👀 Ver una muestra de los datos subidos (Primeras 5 filas)"):
            st.dataframe(df_produccion.head(5), use_container_width=True)

        # ======================================================================
        # 2. PREDICCIÓN MASIVA: CONTEO DE POSIBLES FRAUDES
        # ======================================================================
        st.divider()
        st.subheader("2. Detección Masiva de Posibles Fraudes")
        st.write("Aplicando el modelo a todas las transacciones del archivo...")
        
        # Preprocesamiento rápido del lote
        df_pred = df_produccion.copy()
        
        # Si faltan columnas de feature engineering, las creamos al vuelo
        if "amount_log" not in df_pred.columns:
            df_pred["amount_log"] = np.log1p(df_pred["amount"])
        
        # Seleccionamos las columnas que espera el modelo
        if hasattr(modelo, 'feature_names_in_'):
            columnas_modelo = modelo.feature_names_in_
        elif hasattr(modelo.estimator, 'feature_names_in_'):
            columnas_modelo = modelo.estimator.feature_names_in_
        else:
            columnas_modelo = df_pred.columns
            
        X_batch = df_pred.reindex(columns=columnas_modelo, fill_value=0)
        
        # Predecir probabilidades
        probabilidades = modelo.predict_proba(X_batch)[:, 1]
        
        # Aplicar el umbral financiero (0.12)
        umbral = 0.12
        df_pred["es_fraude_predicho"] = (probabilidades >= umbral).astype(int)
        df_pred["probabilidad_fraude"] = probabilidades
        
        # Contadores
        total_fraudes = df_pred["es_fraude_predicho"].sum()
        monto_en_riesgo = df_pred.loc[df_pred["es_fraude_predicho"] == 1, "amount"].sum() if "amount" in df_pred.columns else 0
        
        # Mostrar tarjetas de resultados
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("🚨 Posibles Fraudes Detectados", f"{total_fraudes:,}")
        col_res2.metric("💰 Monto en Riesgo", f"Bs {monto_en_riesgo:,.2f}")
        col_res3.metric("✅ Transacciones Seguras", f"{len(df_pred) - total_fraudes:,}")
        
        # Mostrar tabla de los fraudes detectados
        if total_fraudes > 0:
            st.warning(f"El modelo ha marcado {total_fraudes} transacciones como sospechosas. Aquí están las más críticas:")
            # Mostrar las 10 con mayor probabilidad
            top_fraudes = df_pred[df_pred["es_fraude_predicho"] == 1].sort_values("probabilidad_fraude", ascending=False).head(10)
            
            # Columnas a mostrar
            cols_mostrar = ["amount", "probabilidad_fraude", "hour_of_day", "type_TRANSFER"]
            cols_mostrar = [c for c in cols_mostrar if c in top_fraudes.columns]
            
            st.dataframe(top_fraudes[cols_mostrar], use_container_width=True)
        else:
            st.success("✅ No se detectaron fraudes en este lote.")

        # ======================================================================
        # 3. EVALUACIÓN DE DRIFT CON EVIDENTLY
        # ======================================================================
        st.divider()
        st.subheader("3. Evaluación de Data Drift")
        
        df_ref_num = df_referencia.select_dtypes(include=[np.number])
        df_prod_num = df_produccion.select_dtypes(include=[np.number])
        columnas_comunes = list(set(df_ref_num.columns) & set(df_prod_num.columns))
        
        reporte = Report(metrics=[DataDriftPreset()])
        evaluacion = reporte.run(reference_data=df_ref_num[columnas_comunes], 
                                 current_data=df_prod_num[columnas_comunes])
        
        try:
            resultado = evaluacion.dict()
        except AttributeError:
            resultado = reporte.as_dict()
        
        drift_detectado = False
        for metric in resultado.get("metrics", []):
            if "result" in metric and "dataset_drift" in metric["result"]:
                drift_detectado = metric["result"]["dataset_drift"]
                break
        
        if drift_detectado:
            st.error("🚨 **ALERTA: Data Drift Detectado**. Los datos han cambiado significativamente. Reentrenar el modelo.")
        else:
            st.success("✅ **Sin Data Drift**. Los datos son estables. El modelo sigue siendo válido.")
        
        # ======================================================================
        # 4. REPORTE VISUAL
        # ======================================================================
        st.divider()
        st.subheader("4. Reporte Visual Detallado (Evidently)")
        
        try:
            ruta_reporte = "reporte_drift_temp.html"
            evaluacion.save_html(ruta_reporte)
            
            with open(ruta_reporte, "r", encoding="utf-8") as f:
                html_data = f.read()
            
            st.download_button(
                label="📥 Descargar Reporte Completo (HTML)",
                data=html_data,
                file_name="reporte_drift_evidently.html",
                mime="text/html"
            )
            
            st.write("---")
            st.info("💡 Puedes hacer scroll dentro de esta ventana para ver el reporte completo.")
            st.components.v1.html(html_data, height=1500, scrolling=True)
            
            if os.path.exists(ruta_reporte):
                os.remove(ruta_reporte)
                
        except Exception as e:
            st.warning(f"No se pudo generar el reporte visual. Error: {e}")
            
    elif archivo_subido is None:
        st.info("Por favor, sube un archivo CSV (ej. `drift_alerta.csv`) para comenzar.")