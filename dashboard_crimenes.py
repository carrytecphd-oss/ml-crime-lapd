
#librerias usadas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib
import warnings
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    PrecisionRecallDisplay, average_precision_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Configuración de la página 
st.set_page_config(
    page_title="LAPD Crime Predictor Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

#funciones de carga de datos y congifuracion para el analisis 

# Ruta del dataset
CARPETA = pathlib.Path(".")
RUTA_DATOS = CARPETA / "datos_dashboard.parquet"

# Columnas que se usarán para el modelo (definición centralizada)
FEATURES_MODELO = [
    'AREA', 'hour', 'month', 'day_of_week', 'is_weekend',
    'Vict Age', 'Premis Cd',
    'has_weapon',
    'is_night',
    'weapon_night',
    'Vict Sex_F', 'Vict Sex_H', 'Vict Sex_M', 'Vict Sex_X',
    'Vict Descent_A', 'Vict Descent_B', 'Vict Descent_C', 'Vict Descent_D',
    'Vict Descent_F', 'Vict Descent_G', 'Vict Descent_H', 'Vict Descent_I',
    'Vict Descent_J', 'Vict Descent_K', 'Vict Descent_L', 'Vict Descent_O',
    'Vict Descent_P', 'Vict Descent_S', 'Vict Descent_U', 'Vict Descent_W',
    'Vict Descent_X', 'Vict Descent_Z',
    'time_block_Madrugada', 'time_block_Mañana', 'time_block_Noche',
    'time_block_Noche_temprana', 'time_block_Tarde'
]

@st.cache_data(ttl=3600)
def cargar_y_preprocesar_datos():
    """Carga el archivo parquet y prepara los datos para el entrenamiento."""
    if not RUTA_DATOS.exists():
        st.error(f"No se encontró el archivo {RUTA_DATOS}. Asegúrate de tenerlo en la misma carpeta.")
        st.stop()

    df = pd.read_parquet(RUTA_DATOS)
    
    #ver que las columnas existan 
    if 'hour' not in df.columns and 'TIME OCC' in df.columns:
        df['hour'] = df['TIME OCC'] // 100
    elif 'hour' not in df.columns:
        df['hour'] = 12
    
    if 'month' not in df.columns and 'DATE OCC' in df.columns:
        df['month'] = pd.to_datetime(df['DATE OCC']).dt.month
    elif 'month' not in df.columns:
        df['month'] = 6
    
    if 'day_of_week' not in df.columns and 'DATE OCC' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['DATE OCC']).dt.dayofweek
    elif 'day_of_week' not in df.columns:
        df['day_of_week'] = 0
    
    if 'is_weekend' not in df.columns and 'day_of_week' in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    elif 'is_weekend' not in df.columns:
        df['is_weekend'] = 0
    
    if 'is_night' not in df.columns and 'hour' in df.columns:
        df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 5)).astype(int)
    elif 'is_night' not in df.columns:
        df['is_night'] = 0
    
    if 'weapon_night' not in df.columns:
        df['weapon_night'] = df.get('has_weapon', 0) * df['is_night']
    
    # Asegurar columnas de dummies
    dummy_cols = {
        'Vict Sex': ['F', 'H', 'M', 'X'],
        'Vict Descent': ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'O', 'P', 'S', 'U', 'W', 'X', 'Z'],
        'time_block': ['Madrugada', 'Mañana', 'Tarde', 'Noche_temprana', 'Noche']
    }
    
    for prefix, categories in dummy_cols.items():
        for cat in categories:
            col_name = f"{prefix}_{cat}"
            if col_name not in df.columns:
                df[col_name] = 0
    
    features_existentes = [f for f in FEATURES_MODELO if f in df.columns]
    
    # Preparación de Datos para Entrenamiento (Split temporal)
    if 'DATE OCC' in df.columns:
        df = df.sort_values('DATE OCC').reset_index(drop=True)
        fecha_corte = df['DATE OCC'].quantile(0.8)
        train = df[df['DATE OCC'] < fecha_corte].copy()
        test = df[df['DATE OCC'] >= fecha_corte].copy()
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.get('target', None))
    
    # Clustering geoespacial
    if 'LAT' in train.columns and 'LON' in train.columns:
        kmeans_geo = MiniBatchKMeans(n_clusters=20, random_state=42, batch_size=10000)
        kmeans_geo.fit(train[['LAT', 'LON']])
        train['geo_cluster'] = kmeans_geo.predict(train[['LAT', 'LON']])
        test['geo_cluster'] = kmeans_geo.predict(test[['LAT', 'LON']])
        features_existentes.append('geo_cluster')
    else:
        kmeans_geo = None
        train['geo_cluster'] = 0
        test['geo_cluster'] = 0
        features_existentes.append('geo_cluster')
    
    # Densidad de crímenes por área
    target_col = 'target' if 'target' in df.columns else 'Part 1-2'
    if target_col == 'Part 1-2':
        train[target_col] = (train['Part 1-2'] == 1).astype(int)
        test[target_col] = (test['Part 1-2'] == 1).astype(int)
    
    density_map = train.groupby('AREA')[target_col].mean()
    train['crime_density_area'] = train['AREA'].map(density_map)
    test['crime_density_area'] = test['AREA'].map(density_map).fillna(density_map.mean())
    features_existentes.append('crime_density_area')
    
    # Separar X e y
    X_train = train[features_existentes].dropna()
    y_train = train.loc[X_train.index, target_col]
    X_test = test[features_existentes].dropna()
    y_test = test.loc[X_test.index, target_col]
    
    return df, X_train, y_train, X_test, y_test, features_existentes, kmeans_geo


@st.cache_resource
def entrenar_modelos(X_train, y_train):
    """Entrena los tres modelos sin mostrar mensajes de progreso."""
    start_time = time.time()
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    
    # 1. Regresión Logística
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced', random_state=42))
    ])
    pipeline_lr.fit(X_train, y_train)
    
    # 2. Random Forest
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=10,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)
    
    # 3. XGBoost
    modelo_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    modelo_xgb.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    return pipeline_lr, modelo_rf, modelo_xgb, elapsed_time

#funciones de visualizacion de graficas
def generar_recomendaciones(prob_grave):
    """Genera recomendaciones operativas basadas en la probabilidad."""
    if prob_grave > 0.6:
        return {
            'nivel': 'CRÍTICO',
            'color': '#ff0000',
            'accion': 'Enviar unidad táctica inmediata - Prioridad ABSOLUTA',
            'recursos': 'Asignar 3-4 patrullas adicionales',
            'tiempo_respuesta': 'Objetivo: < 5 minutos'
        }
    elif prob_grave > 0.35:
        return {
            'nivel': 'ALERTA',
            'color': '#ff8c00',
            'accion': 'Aumentar patrullaje preventivo',
            'recursos': 'Asignar 1-2 patrullas adicionales',
            'tiempo_respuesta': 'Objetivo: < 8 minutos'
        }
    else:
        return {
            'nivel': 'MONITOREO',
            'color': '#00cc00',
            'accion': 'Mantener vigilancia estándar',
            'recursos': 'Recursos normales',
            'tiempo_respuesta': 'Objetivo: < 12 minutos'
        }

def crear_mapa_calor_riesgo(df, col_riesgo='prob_grave'):
    """Crea un mapa de calor geoespacial de riesgo."""
    if 'LAT' not in df.columns or 'LON' not in df.columns:
        return None
    
    sample_size = min(10000, len(df))
    df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    col_to_plot = col_riesgo if col_riesgo in df.columns else 'target'
    
    fig = px.density_mapbox(
        df_sample,
        lat="LAT", lon="LON",
        z=col_to_plot,
        radius=8,
        center={"lat": 34.05, "lon": -118.25},
        zoom=9,
        mapbox_style="carto-positron",
        color_continuous_scale="YlOrRd",
        title="Mapa de Riesgo"
    )
    return fig

def mostrar_metricas_modelo(y_test, y_pred, y_pred_proba, modelo_nombre):
    """Muestra métricas clave del modelo en tarjetas."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precisión", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")
    col5.metric("ROC-AUC", f"{auc:.3f}")

def graficar_comparacion_tipos_por_hora(df_filtrado):
    """Gráfica comparativa de crímenes Tipo 1 vs Tipo 2 por hora."""
    if 'hour' not in df_filtrado.columns or 'target' not in df_filtrado.columns:
        return None
    
    df_hora = df_filtrado.groupby('hour')['target'].agg(['count', 'sum']).reset_index()
    df_hora.columns = ['hour', 'total_crimenes', 'crimenes_tipo1']
    df_hora['crimenes_tipo2'] = df_hora['total_crimenes'] - df_hora['crimenes_tipo1']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_hora['hour'], y=df_hora['crimenes_tipo1'], 
                         name='Tipo 1 (Grave)', marker_color='red'))
    fig.add_trace(go.Bar(x=df_hora['hour'], y=df_hora['crimenes_tipo2'], 
                         name='Tipo 2 (Menor)', marker_color='orange'))
    fig.update_layout(barmode='group', title='Comparación de Tipos de Crimen por Hora',
                      xaxis_title='Hora del día', yaxis_title='Número de incidentes')
    return fig

def graficar_comparacion_tipos_por_area(df_filtrado, area_col='AREA NAME'):
    """Gráfica comparativa de crímenes Tipo 1 vs Tipo 2 por área."""
    if area_col not in df_filtrado.columns or 'target' not in df_filtrado.columns:
        return None
    
    df_area = df_filtrado.groupby(area_col)['target'].agg(['count', 'sum']).reset_index()
    df_area.columns = [area_col, 'total_crimenes', 'crimenes_tipo1']
    df_area['crimenes_tipo2'] = df_area['total_crimenes'] - df_area['crimenes_tipo1']
    df_area = df_area.sort_values('crimenes_tipo1', ascending=False).head(15)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_area[area_col], y=df_area['crimenes_tipo1'], 
                         name='Tipo 1 (Grave)', marker_color='red'))
    fig.add_trace(go.Bar(x=df_area[area_col], y=df_area['crimenes_tipo2'], 
                         name='Tipo 2 (Menor)', marker_color='orange'))
    fig.update_layout(barmode='group', title='Comparación de Tipos de Crimen por Área',
                      xaxis_title='Área', yaxis_title='Número de incidentes',
                      xaxis_tickangle=45)
    return fig

def graficar_comparacion_tipos_por_edad(df_filtrado):
    """Gráfica de distribución de tipos de crimen por edad de la víctima."""
    if 'Vict Age' not in df_filtrado.columns or 'target' not in df_filtrado.columns:
        return None
    
    #rangos de edad
    df_edad = df_filtrado.copy()
    df_edad['rango_edad'] = pd.cut(df_edad['Vict Age'], 
                                    bins=[0, 18, 30, 45, 60, 100], 
                                    labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    
    df_rango = df_edad.groupby('rango_edad', observed=True)['target'].value_counts().unstack().fillna(0)
    df_rango = df_rango.rename(columns={0: 'Tipo 2 (Menor)', 1: 'Tipo 1 (Grave)'})
    
    fig = px.bar(df_rango, barmode='group', title='Comparación de Tipos de Crimen por Edad de la Víctima',
                 labels={'value': 'Número de incidentes', 'rango_edad': 'Rango de Edad'},
                 color_discrete_map={'Tipo 1 (Grave)': 'red', 'Tipo 2 (Menor)': 'orange'})
    return fig

def graficar_comparativa_modelos(y_test, y_pred_lr, y_pred_rf, y_pred_xgb, 
                                  y_prob_lr, y_prob_rf, y_prob_xgb):
    """Gráfica comparativa de los tres modelos en una sola visualización."""
    modelos = ['Regresión Logística', 'Random Forest', 'XGBoost']
    predictions = [y_pred_lr, y_pred_rf, y_pred_xgb]
    probabilities = [y_prob_lr, y_prob_rf, y_prob_xgb]
    
    #métricas de cada modelo
    metricas = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for y_pred, y_prob in zip(predictions, probabilities):
        metricas['accuracy'].append(accuracy_score(y_test, y_pred))
        metricas['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metricas['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metricas['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metricas['auc'].append(roc_auc_score(y_test, y_prob))
    
    # gráfico de barras para los modelos
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, modelo in enumerate(modelos):
        fig.add_trace(go.Bar(
            name=modelo,
            x=list(metricas.keys()),
            y=[metricas['accuracy'][i], metricas['precision'][i], 
               metricas['recall'][i], metricas['f1'][i], metricas['auc'][i]],
            text=[f'{v:.2%}' for v in [metricas['accuracy'][i], metricas['precision'][i], 
                                        metricas['recall'][i], metricas['f1'][i], metricas['auc'][i]]],
            textposition='auto',
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title='Comparación de Rendimiento de Modelos',
        xaxis_title='Métrica',
        yaxis_title='Valor',
        yaxis_tickformat='.0%',
        barmode='group',
        legend_title='Modelo',
        height=500
    )
    return fig

#interfaz y panel de control

#Carga de Datos y Modelos
with st.sidebar:
    st.title("Panel de Control")
    st.markdown("---")
    
    with st.status("Cargando datos y entrenando modelos...", expanded=False) as status:
        df_original, X_train, y_train, X_test, y_test, FEATURES, kmeans_geo = cargar_y_preprocesar_datos()
        modelo_lr, modelo_rf, modelo_xgb, tiempo_entrenamiento = entrenar_modelos(X_train, y_train)
        status.update(label="¡Sistema listo!", state="complete")
    
    st.success(f"Modelo listo")
    
    st.subheader("Filtros Geográficos")
    
    #tipo de área con nombres
    if 'AREA NAME' in df_original.columns:
        areas_disponibles = sorted(df_original['AREA NAME'].dropna().unique())
    elif 'AREA' in df_original.columns and 'AREA NAME' not in df_original.columns:
        area_mapping = {1: 'Central', 2: 'Rampart', 3: 'Southwest', 4: 'Hollenbeck', 5: 'Harbor',
                        6: 'Hollywood', 7: 'Wilshire', 8: 'West LA', 9: 'Van Nuys', 10: 'West Valley',
                        11: 'Northeast', 12: '77th Street', 13: 'Newton', 14: 'Pacific', 15: 'N Hollywood',
                        16: 'Foothill', 17: 'Devonshire', 18: 'Southeast', 19: 'Mission', 20: 'Olympic',
                        21: 'Topanga'}
        df_original['AREA NAME'] = df_original['AREA'].map(area_mapping)
        areas_disponibles = sorted(df_original['AREA NAME'].dropna().unique())
    else:
        areas_disponibles = [f"Área {i}" for i in range(1, 22)]

    areas_seleccionadas = st.multiselect(
        "Áreas",
        options=areas_disponibles,
        default=areas_disponibles[:2] if len(areas_disponibles) > 2 else areas_disponibles
    )

    st.subheader("Configuración")
    umbral_riesgo = st.slider("Umbral de Riesgo Grave", 0.0, 1.0, 0.35, 0.05)

#Filtros en los Datos
df_filtrado = df_original.copy()
if areas_seleccionadas and 'AREA NAME' in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado['AREA NAME'].isin(areas_seleccionadas)]

# Añadir predicciones
try:
    df_features_filtrado = pd.DataFrame(index=df_filtrado.index)
    for col in FEATURES:
        if col in df_filtrado.columns:
            df_features_filtrado[col] = df_filtrado[col]
        else:
            df_features_filtrado[col] = 0
    df_filtrado['prob_grave'] = modelo_rf.predict_proba(df_features_filtrado)[:, 1]
    df_filtrado['prediccion'] = (df_filtrado['prob_grave'] >= umbral_riesgo).astype(int)
except Exception as e:
    st.warning(f"No se pudieron generar predicciones para el filtro actual: {e}")
    df_filtrado['prob_grave'] = 0.5
    df_filtrado['prediccion'] = 0

# inicio de pagina 
col_logo, col_title = st.columns([1, 3])
with col_logo:
     st.image("logo.jpeg", width=350,)
with col_title:
    st.title("LAPD Crime Predictor")
    st.markdown("""
    <div style='background-color: #1e3a8a; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>Sistema de Apoyo a la Toma de Decisiones</h3>
    <p style='color: #e0e7ff; margin: 0.5rem 0 0 0;'>
    Modelos de Machine Learning para optimizar la asignación de recursos segun el tipo de delitos.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# indicadores clave

st.header("Indicadores Clave")
col1, col2, col3, col4, col5 = st.columns(5)

total_incidentes = len(df_filtrado)
col1.metric("Total Incidentes", f"{total_incidentes:,}")

if 'target' in df_filtrado.columns:
    graves = df_filtrado['target'].sum()
    pct_graves = graves / total_incidentes if total_incidentes > 0 else 0
    col2.metric("Crímenes Graves (Tipo 1)", f"{graves:,}", delta=f"{pct_graves:.1%}")
else:
    col2.metric("Crímenes Graves", "N/A")

riesgo_promedio = df_filtrado['prob_grave'].mean()
col3.metric("Riesgo Promedio", f"{riesgo_promedio:.1%}")

if 'AREA NAME' in df_filtrado.columns:
    area_riesgo = df_filtrado.groupby('AREA NAME')['prob_grave'].mean().idxmax()
    col4.metric("Área Mayor Riesgo", area_riesgo[:20] + "..." if len(area_riesgo) > 20 else area_riesgo)
else:
    col4.metric("Área Mayor Riesgo", "N/A")

if 'hour' in df_filtrado.columns:
    hora_pico = df_filtrado.groupby('hour')['prob_grave'].mean().idxmax()
    col5.metric("Hora Más Riesgosa", f"{int(hora_pico):02d}:00")
else:
    col5.metric("Hora Más Riesgosa", "N/A")

st.markdown("---")

#graficas de las comparaciones de los modelos
st.header("Rendimiento de los modelos")

# Predicciones para evaluación
y_pred_lr = modelo_lr.predict(X_test)
y_prob_lr = modelo_lr.predict_proba(X_test)[:, 1]

y_pred_rf = modelo_rf.predict(X_test)
y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]

y_pred_xgb = modelo_xgb.predict(X_test)
y_prob_xgb = modelo_xgb.predict_proba(X_test)[:, 1]

# Gráfica comparativa de los tres modelos
fig_comparativa = graficar_comparativa_modelos(
    y_test, y_pred_lr, y_pred_rf, y_pred_xgb,
    y_prob_lr, y_prob_rf, y_prob_xgb
)
st.plotly_chart(fig_comparativa, use_container_width=True)

# Expandir para ver matrices de confusión 
with st.expander("Matrices de Confusión de los modelos"):
       
    tab1, tab2, tab3 = st.tabs(["Regresión Logística", "Random Forest", "XGBoost"])
    
    with tab1:
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig_cm_lr = px.imshow(cm_lr, text_auto=True, color_continuous_scale='Blues',
                              labels=dict(x="Predicción", y="Real", color="Conteo"),
                              title="Matriz de Confusión - Regresión Logística")
        st.plotly_chart(fig_cm_lr, use_container_width=True)
    
    with tab2:
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig_cm_rf = px.imshow(cm_rf, text_auto=True, color_continuous_scale='Blues',
                              labels=dict(x="Predicción", y="Real", color="Conteo"),
                              title="Matriz de Confusión - Random Forest")
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        
        st.subheader("Características importantes")
        importancias = pd.Series(modelo_rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig_imp = px.bar(importancias.tail(10), orientation='h',
                         color=importancias.tail(10).values, color_continuous_scale='Reds',
                         title="Top 10",
                         labels={'value': 'Importancia', 'index': 'Característica'})
        st.plotly_chart(fig_imp, use_container_width=True)
        

    
    with tab3:
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        fig_cm_xgb = px.imshow(cm_xgb, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicción", y="Real", color="Conteo"),
                               title="Matriz de Confusión - XGBoost")
        st.plotly_chart(fig_cm_xgb, use_container_width=True)



#Mostrar graficas de tip 1 vs tipo 2

st.header("Análisis Comparativo: Tipo 1 (Grave) vs Tipo 2 (Menor)")
st.markdown("Usar filtros geograficos del panel de control ")

col_hora, col_area = st.columns(2)

with col_hora:
    fig_hora = graficar_comparacion_tipos_por_hora(df_filtrado)
    if fig_hora:
        st.plotly_chart(fig_hora, use_container_width=True)
    else:
        st.info("Datos insuficientes para gráfico por hora")

with col_area:
    fig_area = graficar_comparacion_tipos_por_area(df_filtrado)
    if fig_area:
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("Datos insuficientes para gráfico por área")

st.plotly_chart(graficar_comparacion_tipos_por_edad(df_filtrado), use_container_width=True)


# visualisacion mapa y horas

st.header("Análisis Geográfico y Temporal")

col_mapa, col_horario = st.columns(2)

with col_mapa:
    fig_mapa = crear_mapa_calor_riesgo(df_filtrado, 'prob_grave')
    if fig_mapa is not None:
        st.plotly_chart(fig_mapa, use_container_width=True)
    else:
        st.info("Datos geográficos no disponibles para visualización en mapa.")

with col_horario:
    if 'hour' in df_filtrado.columns and 'prob_grave' in df_filtrado.columns:
        riesgo_horario = df_filtrado.groupby('hour')['prob_grave'].mean().reset_index()
        fig_horario = px.bar(
            riesgo_horario, x='hour', y='prob_grave', color='prob_grave',
            color_continuous_scale='Reds',
            title="Riesgo Promedio por Hora del Día",
            labels={'hour': 'Hora', 'prob_grave': 'Probabilidad de Crimen Grave'}
        )
        fig_horario.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig_horario, use_container_width=True)
    else:
        st.info("Datos horarios no disponibles para análisis.")

# Clustering Geoespacial
st.subheader("Clustering con zonas de riesgo")
if kmeans_geo is not None and 'LAT' in df_filtrado.columns and 'LON' in df_filtrado.columns:
    coords = df_filtrado[['LAT', 'LON']].dropna()
    if len(coords) > 0:
        clusters = kmeans_geo.predict(coords)
        df_filtrado['cluster'] = clusters
        
        fig_cluster = px.scatter_mapbox(
            df_filtrado.sample(min(5000, len(df_filtrado)), random_state=42),
            lat="LAT", lon="LON", color="cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
            size_max=5, zoom=9, center={"lat": 34.05, "lon": -118.25},
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        if 'target' in df_filtrado.columns:
            cluster_stats = df_filtrado.groupby('cluster').agg(
                total_incidentes=('target', 'count'),
                crimenes_graves=('target', 'sum'),
                porcentaje_graves=('target', 'mean')
            ).reset_index()
            st.subheader("Estadísticas por Zona de Riesgo")
            st.dataframe(cluster_stats.style.format({'porcentaje_graves': '{:.1%}'}), use_container_width=True)
    else:
        st.warning("No hay suficientes datos con coordenadas para realizar clustering.")
else:
    st.warning("El dataset no contiene coordenadas LAT/LON para el clustering geoespacial.")


# Decisiones y recomentacion 

st.header("Decisiones y Recomendaciones Operativas")

if 'AREA NAME' in df_filtrado.columns and 'prob_grave' in df_filtrado.columns:
    areas_criticas = df_filtrado.groupby('AREA NAME')['prob_grave'].mean().sort_values(ascending=False).head(5)
    
    for area, riesgo in areas_criticas.items():
        rec = generar_recomendaciones(riesgo)
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div style='background-color: {rec['color']}20; padding: 1rem; border-radius: 10px; border-left: 5px solid {rec['color']};'>
                    <h4 style='margin: 0;'>{area}</h4>
                    <p style='margin: 0; font-size: 24px; font-weight: bold;'>{riesgo:.1%}</p>
                    <p style='margin: 0; font-size: 12px;'>Riesgo Grave</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                - **Nivel:** 🔴 **{rec['nivel']}**
                - **Acción:** {rec['accion']}
                - **Recursos:** {rec['recursos']}
                - **Tiempo Respuesta:** {rec['tiempo_respuesta']}
                """)
        st.markdown("---")

st.subheader("Recomendaciones")
col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.success("""
    **Asignación de Recursos**
    
    - **Áreas con mayor riesgo**: Aumentar patrullajes en 40% durante horario nocturno (10pm-4am)
    - **Zonas comerciales**: Implementar 2 unidades adicionales en fines de semana
    - **Zona centrica**: implementar vigilancia peatonal           
    - **Áreas residenciales**: Mantener vigilancia actual con enfoque preventivo
    """)

with col_rec2:
    st.info("""
    **Plan Estratégico**
    
    - Rotar recursos desde áreas de bajo riesgo hacia zonas críticas
    - Implementar sistema de alerta temprana basado en predicciones diarias
    - Entrenar oficiales para las áreas de alto riesgo
    - Evaluar cada 2 semanas y ajustar el modelo si lo requiere 
    """)


# Predictor (randomForest)

st.header("Predictor de Severidad (Random Forest)")

with st.expander("Ingrese los datos del incidente para obtener una predicción", expanded=True):

    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Usar nombre de área si está disponible
        if 'AREA NAME' in df_original.columns:
            areas_list = sorted(df_original['AREA NAME'].dropna().unique())
            area_seleccionada = st.selectbox("Área", options=areas_list, 
                                            help="Distrito donde ocurre el incidente")
            # Mapear nombre a código
            if 'AREA' in df_original.columns:
                area_mapping_rev = {v: k for k, v in df_original.set_index('AREA NAME')['AREA'].to_dict().items()}
                area_in = area_mapping_rev.get(area_seleccionada, 1)
            else:
                area_in = areas_list.index(area_seleccionada) + 1 if area_seleccionada else 1
        else:
            area_in = st.number_input("Área (código 1-21)", min_value=1, max_value=21, value=1,
                                     help="Código del distrito policial")
        
        hora_in = st.number_input("Hora del día (0-23)", min_value=0, max_value=23, value=12,
                                  help="Hora en que ocurre el incidente (formato 24h)")
        edad_in = st.number_input("Edad de la víctima", min_value=0, max_value=100, value=30,
                                  help="Edad de la víctima en años")
    
    with col2:
        arma_in = st.selectbox("¿Hubo uso de arma?", options=[0, 1], 
                               format_func=lambda x: "Sí" if x == 1 else "No",
                               help="1 = si, 0 = no ")
        mes_in = st.number_input("Mes (1-12)", min_value=1, max_value=12, value=6,
                                help="Mes del año en que ocurre el incidente")
        dia_semana = st.number_input("Día de la semana (0=Lunes, 6=Domingo)", min_value=0, max_value=6, value=0,
                                     help="Día de la semana del incidente")
    
    with col3:
        premis_in = st.number_input("Código de lugar (Premis Cd)", min_value=0, max_value=999, value=101,
                                    help="""Código que identifica el tipo de lugar donde ocurre el crimen:
- 101: Calle/ACERA
- 102: Estacionamiento
- 103: Residencia/Vivienda
- 104: Comercio/Restaurante
- 105: Escuela
- 106: Parque/Área recreativa
- 107: Transporte público
- 108: Oficina/Banco
- 109: Hotel/Motel
- 110: Hospital
- 111: Iglesia
- 112: Estacionamiento
- 113: Banco/Cajero automático
- 114: Tienda de conveniencia
- 115: Bar/Club nocturno
- 116: Gasolinera
- 117: Construcción
- 118: Edificio gubernamental
- 119: Parque industrial
- 120: Área de juegos""")
        
        
    
    if st.button("Predecir Severidad", type="primary"):
        input_data = {col: 0 for col in FEATURES}
        input_data['AREA'] = area_in
        input_data['hour'] = hora_in
        input_data['month'] = mes_in
        input_data['day_of_week'] = dia_semana
        input_data['is_weekend'] = 1 if dia_semana >= 5 else 0
        input_data['Vict Age'] = edad_in
        input_data['Premis Cd'] = premis_in
        input_data['has_weapon'] = arma_in
        input_data['is_night'] = 1 if (hora_in >= 20 or hora_in <= 5) else 0
        input_data['weapon_night'] = arma_in * input_data['is_night']
        
        input_df = pd.DataFrame([input_data])
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0
        
        prob = modelo_rf.predict_proba(input_df[FEATURES])[0, 1]
        nivel = "🔴 GRAVE (Tipo 1)" if prob >= umbral_riesgo else "🟠 MENOR (Tipo 2)"
        
        st.success(f"### Clasificación Predicha: {nivel}")
        st.metric("Probabilidad de Crimen Grave", f"{prob:.1%}")
        
        # Mostrar qué factores influyeron más en la predicción
        st.markdown("#### Factores de riesgo identificados:")
        factores = []
        if arma_in == 1:
            factores.append("Presencia de arma (factor de alto riesgo)")
        if hora_in >= 20 or hora_in <= 5:
            factores.append("Horario nocturno/madrugada (mayor riesgo)")
        if dia_semana >= 5:
            factores.append("Fin de semana (incremento de riesgo)")
        if edad_in < 18 or edad_in > 65:
            factores.append("Víctima en rango de edad vulnerable")
        if premis_in in [101, 115, 116]:
            factores.append("Lugar de alto riesgo (calle/bar/gasolinera)")
        
        if factores:
            for factor in factores:
                st.write(factor)
        else:
            st.write("No se identificaron factores de alto riesgo")
        
        rec = generar_recomendaciones(prob)
        st.info(f"""
        **Recomendación Operativa:**
        - **Nivel:** {rec['nivel']}
        - **Acción:** {rec['accion']}
        - **Recursos:** {rec['recursos']}
        - **Tiempo de respuesta objetivo:** {rec['tiempo_respuesta']}
        """)