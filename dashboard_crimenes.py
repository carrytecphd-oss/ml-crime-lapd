import streamlit as st
import pandas as pd
import plotly.express as px
import pickle, pathlib

CARPETA = pathlib.Path(__file__).parent

st.set_page_config(page_title="LAPD Crime Dashboard", layout="wide", page_icon=":bar_chart:")

DRIVE_MODEL_ID = "1N7Fad4W2UmJJqoTmhLhw6mSXvUhv5K3n"  

@st.cache_data
def cargar_datos():
    return pd.read_parquet(CARPETA / "datos_dashboard.parquet")

@st.cache_resource
def cargar_modelo():
    model_path = CARPETA / "modelo_rf.pkl"
    if not model_path.exists():
        import gdown
        gdown.download(id=DRIVE_MODEL_ID, output=str(model_path), quiet=False)
    with open(model_path, "rb") as f:
        return pickle.load(f)

df     = cargar_datos()
modelo = cargar_modelo()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
areas       = st.sidebar.multiselect("Área", sorted(df["area_name"].dropna().unique()),
                                     default=sorted(df["area_name"].dropna().unique())[:5])
horas       = st.sidebar.slider("Rango de horas", 0, 23, (0, 23))
solo_graves = st.sidebar.checkbox("Solo crímenes graves (Tipo 1)", value=False)

mask = df["area_name"].isin(areas) & df["hour"].between(horas[0], horas[1])
if solo_graves:
    mask &= (df["target"] == 1)
dff = df[mask]

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.title("Dashboard Ejecutivo — Crímenes LAPD 2020-2024")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total incidentes",         f"{len(dff):,}")
k2.metric("Crímenes graves (Tipo 1)", f"{dff['target'].sum():,}")
k3.metric("% Crímenes graves",        f"{dff['target'].mean():.1%}")
k4.metric("Prob. promedio (modelo)",  f"{dff['prob_grave'].mean():.1%}")

st.divider()

# ── Gráficos ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Crímenes por Hora")
    hora_df = dff.groupby(["hour","target"]).size().reset_index(name="n")
    hora_df["Tipo"] = hora_df["target"].map({1:"Tipo 1 (grave)", 0:"Tipo 2 (menor)"})
    fig = px.bar(hora_df, x="hour", y="n", color="Tipo",
                 color_discrete_map={"Tipo 1 (grave)":"tomato","Tipo 2 (menor)":"steelblue"},
                 labels={"hour":"Hora", "n":"Incidentes"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Distribución por Área")
    area_df = dff.groupby("area_name")["target"].agg(["sum","count"]).reset_index()
    area_df.columns = ["Area","Graves","Total"]
    area_df["Pct_grave"] = area_df["Graves"] / area_df["Total"]
    fig2 = px.bar(area_df.sort_values("Pct_grave", ascending=False),
                  x="Area", y="Pct_grave",
                  labels={"Pct_grave":"% Crímenes graves"},
                  color="Pct_grave", color_continuous_scale="Reds")
    fig2.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

# ── Mapa de calor ─────────────────────────────────────────────────────────────
st.subheader("Mapa de Riesgo — Probabilidad de Crimen Grave")
fig_map = px.density_mapbox(
    dff.sample(min(10000, len(dff)), random_state=42),
    lat="LAT", lon="LON", z="prob_grave",
    radius=8, center={"lat": 34.05, "lon": -118.25}, zoom=9,
    mapbox_style="carto-positron", color_continuous_scale="YlOrRd"
)
st.plotly_chart(fig_map, use_container_width=True)

# ── Predictor individual ──────────────────────────────────────────────────────
st.subheader("Predictor de Severidad")
with st.form("predictor"):
    c1, c2, c3 = st.columns(3)
    area_in   = c1.number_input("Área (1-21)",    1, 21, 1)
    hora_in   = c2.number_input("Hora (0-23)",    0, 23, 12)
    edad_in   = c3.number_input("Edad víctima",   0, 99, 30)
    arma_in   = c1.selectbox("¿Hubo arma?",       [0, 1])
    premis_in = c2.number_input("Código de lugar",0, 999, 101)
    mes_in    = c3.number_input("Mes (1-12)",      1, 12, 6)
    submitted = st.form_submit_button("Predecir")
    if submitted:
        import pandas as pd
        row = pd.DataFrame([{
            "AREA":area_in, "hour":hora_in, "month":mes_in,
            "day_of_week":0, "is_weekend":0,
            "Vict Age":edad_in, "sex_enc":2, "desc_enc":4,
            "Premis Cd":premis_in, "has_weapon":arma_in,
            "block_enc":2, "LAT":34.05, "LON":-118.25
        }])
        prob  = modelo.predict_proba(row)[0, 1]
        nivel = "🔴 GRAVE (Tipo 1)" if prob >= 0.5 else "🟡 Menor (Tipo 2)"
        st.metric("Clasificación predicha", nivel, f"Probabilidad: {prob:.1%}")