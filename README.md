# LAPD Crime Severity Predictor

### Machine Learning para la clasificación de incidentes criminales en Los Ángeles (2020–2024)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)


---

## Descripción general

Este proyecto aplica técnicas de **aprendizaje automático supervisado** para predecir la severidad de los incidentes criminales registrados por el Departamento de Policía de Los Ángeles (LAPD). A partir de variables como —hora, área, tipo de lugar, presencia de arma y características de la víctima— el sistema clasifica cada evento como:

-  **Tipo 1 — Crimen Grave**
-  **Tipo 2 — Crimen Menor**

El flujo completo cubre análisis exploratorio, ingeniería de variables, entrenamiento comparativo de modelos y un **dashboard interactivo** con predictor individual en tiempo real.

---

## Estructura del repositorio

```
ml-crime-lapd/
│
├── ML_Trabajo_Final.ipynb        # Notebook principal: EDA, preprocesamiento y modelado
├── dashboard_crimenes.py         # Aplicación Streamlit (dashboard ejecutivo)
├── datos_dashboard.parquet       # Dataset procesado listo para producción
├── modelo_rf.pkl                 # Modelo Random Forest serializado
├── requirements.txt              # Dependencias del proyecto
└── Proyecto Final Machine Learning.pdf  # Documentación del proyecto
```

---

## Problema y enfoque

Los departamentos de seguridad pública asignan recursos de manera reactiva porque los sistemas de registro tradicionales **no tienen capacidad predictiva**: catalogan lo ocurrido pero no anticipan la severidad de un evento futuro.

Este proyecto responde a la pregunta:

> *¿Es posible predecir, con precisión aceptable, si un incidente del LAPD será grave o menor a partir de sus variables contextuales?*

El enfoque adoptado es **clasificación binaria supervisada**, con énfasis en reproducibilidad, interpretabilidad operacional y despliegue inmediato.

---


## Metodología

### 1. Preprocesamiento e ingeniería de variables

A partir del dataset crudo del LAPD se construyeron las siguientes features:

| Variable | Descripción |
|---|---|
| `AREA` | División policial (1–21) |
| `hour` | Hora del incidente (0–23) |
| `month` | Mes del reporte (1–12) |
| `day_of_week` | Día de la semana codificado |
| `is_weekend` | Flag binaria: 1 si fue en fin de semana |
| `Vict Age` | Edad de la víctima |
| `sex_enc` | Género de la víctima codificado |
| `desc_enc` | Tipo de crimen codificado |
| `Premis Cd` | Código del tipo de lugar |
| `has_weapon` | Flag binaria: 1 si hubo arma reportada |
| `block_enc` | Codificación del bloque geográfico |
| `LAT` / `LON` | Coordenadas geográficas del evento |

### 2. Modelos evaluados

Se entrenaron tres clasificadores con **validación cruzada **:

| Modelo | Accuracy | AUC-ROC |
|---|---|---|
| Regresión Logística | ~72% | ~0.75 |
| Árbol de Decisión | ~76% | ~0.77 |
| **Random Forest** ✅ | **~83%** | **~0.88** |

El **Random Forest** fue seleccionado como modelo final por su superior capacidad discriminatoria, especialmente relevante para equilibrar la detección de crímenes graves (recall) con la precisión global.

---

## Dashboard ejecutivo

El archivo `dashboard_crimenes.py` implementa un dashboard:

```
┌────────────────────────────────────────────────────────────┐
│  KPIs  │ Total incidentes · Graves · % Graves · Prob. avg  │
├─────────────────────────┬──────────────────────────────────┤
│  Crímenes por hora      │  Distribución por área (LAPD)    │
│  (barras apiladas)      │  (% crímenes graves por división)│
├─────────────────────────┴──────────────────────────────────┤
│  Mapa de densidad de riesgo (LAT/LON · escala YlOrRd)      │
├────────────────────────────────────────────────────────────┤
│  Predictor individual  → ingresa contexto → obtén prob.    │
└────────────────────────────────────────────────────────────┘
```

Los filtros de la barra lateral permiten segmentar por **área LAPD**, **rango horario** y **tipo de crimen**, actualizando todos los componentes en tiempo real.

---

## Instalación y uso

### Prerrequisitos

- Python 3.10 o superior
- pip

### 1. Clonar el repositorio

```bash
git clone https://github.com/carrytecphd-oss/ml-crime-lapd.git
cd ml-crime-lapd
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **requirements.txt** incluye: `gdown`, `pandas`, `plotly`, `pyarrow`, `scikit-learn`, `streamlit`

### 3. Ejecutar el dashboard

```bash
streamlit run dashboard_crimenes.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`.

> **Nota:** si `modelo_rf.pkl` no está presente localmente, el dashboard lo descarga desde Google Drive.

### 4. Explorar el notebook de modelado

Abre `ML_Trabajo_Final.ipynb` con Jupyter Lab o Jupyter Notebook para revisar todo el flujo de EDA, preprocesamiento y entrenamiento:

```bash
jupyter lab ML_Trabajo_Final.ipynb
```

---

### `dashboard_crimenes.py`

Aplicación Streamlit que consume `datos_dashboard.parquet` y `modelo_rf.pkl`. Implementa:
- Carga cacheada de datos (`@st.cache_data`) y modelo (`@st.cache_resource`)
- Descarga automática del modelo desde Google Drive si no existe localmente
- Filtros interactivos por área, hora y tipo de crimen
- Mapa de densidad geoespacial sobre Los Ángeles
- Formulario de predicción individual con salida probabilística

### `ML_Trabajo_Final.ipynb`

Notebook documentado:
- Carga y exploración del dataset del LAPD
- Transformaciones y feature engineering
- Entrenamiento y evaluación comparativa de modelos
- Exportación del modelo final

### `datos_dashboard.parquet`

Dataset procesado en formato Parquet con las columnas necesarias para el dashboard, incluyendo las probabilidades del modelo Random Forest.

---

## Resultados principales

- **Accuracy global:** ~83% (Random Forest)
- **AUC-ROC:** ~0.88
- **Variable más discriminante:** presencia de arma (`has_weapon`)
- **Patrón horario:** concentración de crímenes graves entre 22:00–04:00 h y 16:00–19:00 h
- **Heterogeneidad geográfica:** algunas divisiones del LAPD superan el 40% de crímenes graves sobre el total de incidentes

---

## Recomendaciones

**Asignación de Recursos**
- Áreas con mayor riesgo: Aumentar patrullajes en 40% durante horario nocturno
(10pm-4am)
- Zonas comerciales: Implementar 2 unidades adicionales en fines de semana
- Zona centrica: implementar vigilancia peatonal
- Áreas residenciales: Mantener vigilancia actual con enfoque preventivo

**Plan Estratégico**
- Rotar recursos desde áreas de bajo riesgo hacia zonas críticas
- Implementar sistema de alerta temprana basado en predicciones diarias
- Entrenar oficiales para las áreas de alto riesgo
- Evaluar cada 2 semanas y ajustar el modelo si lo requiere

---
