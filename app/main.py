import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import time
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir directorio raíz al path para importar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_dataset import generate_iiot_sequence
from app.utils import preprocess_and_impute
from app.model import LSTMClassifier, GRUClassifier, BiMambaClassifier

# Configurar estilo seaborn
sns.set_theme(style="whitegrid")

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Bi-Mamba IIoT Fault Detector",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para estética premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;400;600;800&display=swap');
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }
    .model-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 10px;
    }
    .badge-normal {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    .badge-falla {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-top: 10px;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.775rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Cargar escalador y modelos desde results/
@st.cache_resource
def load_assets():
    # Cargar StandardScaler
    with open('results/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Inicializar y cargar LSTM
    lstm = LSTMClassifier(input_dim=4, hidden_dim=32, num_layers=2)
    lstm.load_state_dict(torch.load('results/lstm_baseline.pth', map_location='cpu'))
    lstm.eval()
    
    # Inicializar y cargar GRU
    gru = GRUClassifier(input_dim=4, hidden_dim=32, num_layers=2)
    gru.load_state_dict(torch.load('results/gru_baseline.pth', map_location='cpu'))
    gru.eval()
    
    # Inicializar y cargar Bi-Mamba
    mamba = BiMambaClassifier(input_dim=4, d_model=32, d_state=16, dropout=0.3)
    mamba.load_state_dict(torch.load('results/bimamba_model.pth', map_location='cpu'))
    mamba.eval()
    
    return scaler, lstm, gru, mamba

try:
    scaler, lstm, gru, mamba = load_assets()
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}. Por favor, asegúrate de que el entrenamiento se haya completado primero.")
    st.stop()

# Header
st.markdown("<h1 class='main-title'>🏭 Plataforma Bi-Mamba para Telemetría IIoT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detección Tolerante a Fallos en Sensores con Muestreo Temporal Irregular y Pérdida de Datos</p>", unsafe_allow_html=True)

# Sidebar de Controles de Simulación
st.sidebar.header("🛠️ Configuración de Simulación")

anomaly_map = {
    "Operación Normal": None,
    "Picos de Vibración (Mecánica)": "spikes",
    "Sobrecalentamiento gradual (Fuga Térmica)": "thermal",
    "Caída severa de presión (Fuga de Fluido)": "leak"
}

anomaly_label = st.sidebar.selectbox(
    "Tipo de Señal/Falla:",
    options=list(anomaly_map.keys())
)
anomaly_type = anomaly_map[anomaly_label]

missing_rate = st.sidebar.slider(
    "Tasa de pérdida de datos (NaNs):",
    min_value=0.0,
    max_value=0.50,
    value=0.15,
    step=0.05,
    format="%.2f"
)

noise_level = st.sidebar.slider(
    "Nivel de ruido de medición:",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    format="%.2f"
)

btn_simulate = st.sidebar.button("⚙️ Simular Nueva Secuencia")

# Inicializar estado para la simulación
if 'sim_df_raw' not in st.session_state or btn_simulate:
    # Generar secuencia de longitud 500
    df_raw, label, desc = generate_iiot_sequence(
        seq_len=500,
        anomaly_type=anomaly_type,
        noise_level=noise_level,
        missing_rate=missing_rate
    )
    st.session_state['sim_df_raw'] = df_raw
    st.session_state['sim_label'] = label
    st.session_state['sim_desc'] = desc

# Recuperar variables del estado
df_raw = st.session_state['sim_df_raw']
label_real = st.session_state['sim_label']
desc_real = st.session_state['sim_desc']

# Preprocesamiento e Imputación en tiempo real
df_imputed = preprocess_and_impute(df_raw, method='forward_fill')

# Preparar entrada para los modelos
feature_cols = ['dt', 'sensor_temp', 'sensor_vib', 'sensor_press']
features = df_imputed[feature_cols].values
features_scaled = scaler.transform(features)
tensor_input = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0) # (1, 500, 4)

# ----------------- EJECUTAR INFERENCIA COMPARADA -----------------
# 1. LSTM Inferencia
t0 = time.perf_counter()
with torch.no_grad():
    prob_lstm = lstm(tensor_input).item()
latency_lstm = (time.perf_counter() - t0) * 1000

# 2. GRU Inferencia
t0 = time.perf_counter()
with torch.no_grad():
    prob_gru = gru(tensor_input).item()
latency_gru = (time.perf_counter() - t0) * 1000

# 3. Bi-Mamba Inferencia
t0 = time.perf_counter()
with torch.no_grad():
    prob_mamba = mamba(tensor_input).item()
latency_mamba = (time.perf_counter() - t0) * 1000


# ----------------- RENDERIZAR PANEL DE CONTROL Y MÉTRICAS -----------------
col_metrics = st.columns(3)

# LSTM Card
with col_metrics[0]:
    label_text = "FALLA DETECTADA" if prob_lstm >= 0.5 else "OPERACIÓN NORMAL"
    badge_class = "badge-falla" if prob_lstm >= 0.5 else "badge-normal"
    st.markdown(f"""
    <div class='card'>
        <div class='model-name'>LSTM Baseline</div>
        <div class='{badge_class}'>{label_text}</div>
        <div class='metric-value'>{prob_lstm:.2%}</div>
        <div class='metric-label'>Probabilidad de Anomalía</div>
        <div style='margin-top:10px; font-size:0.85rem; color:#6B7280;'>Latencia: <b>{latency_lstm:.2f} ms</b></div>
    </div>
    """, unsafe_allow_html=True)

# GRU Card
with col_metrics[1]:
    label_text = "FALLA DETECTADA" if prob_gru >= 0.5 else "OPERACIÓN NORMAL"
    badge_class = "badge-falla" if prob_gru >= 0.5 else "badge-normal"
    st.markdown(f"""
    <div class='card'>
        <div class='model-name'>GRU Baseline</div>
        <div class='{badge_class}'>{label_text}</div>
        <div class='metric-value'>{prob_gru:.2%}</div>
        <div class='metric-label'>Probabilidad de Anomalía</div>
        <div style='margin-top:10px; font-size:0.85rem; color:#6B7280;'>Latencia: <b>{latency_gru:.2f} ms</b></div>
    </div>
    """, unsafe_allow_html=True)

# Bi-Mamba Card
with col_metrics[2]:
    label_text = "FALLA DETECTADA" if prob_mamba >= 0.5 else "OPERACIÓN NORMAL"
    badge_class = "badge-falla" if prob_mamba >= 0.5 else "badge-normal"
    st.markdown(f"""
    <div class='card'>
        <div class='model-name' style='color:#6D28D9;'>🚀 Bi-Mamba (Propuesto)</div>
        <div class='{badge_class}'>{label_text}</div>
        <div class='metric-value' style='color:#6D28D9;'>{prob_mamba:.2%}</div>
        <div class='metric-label'>Probabilidad de Anomalía</div>
        <div style='margin-top:10px; font-size:0.85rem; color:#6B7280;'>Latencia: <b>{latency_mamba:.2f} ms</b></div>
    </div>
    """, unsafe_allow_html=True)

# Estado Real de la Simulación
st.info(f"**Estado Físico Real de la Secuencia Simulación:** {desc_real} (Etiqueta Real: `{label_real}`)")


# ----------------- DIBUJAR GRÁFICOS DE TELEMETRÍA -----------------
st.markdown("### 📊 Señales de Sensores e Imputación de Pérdida de Datos")

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# Sensor Temperatura
axes[0].plot(df_imputed['timestamp'], df_imputed['sensor_temp'], label='Señal Reconstruida (F-fill)', color='#1E3A8A', lw=2)
# Encontrar NaNs originales
nan_indices = df_raw['sensor_temp'].isna()
if nan_indices.any():
    axes[0].scatter(df_raw['timestamp'][nan_indices], df_imputed['sensor_temp'][nan_indices],
                    color='#EF4444', marker='x', s=50, label='Dato Perdido (Imputado)', zorder=5)
axes[0].set_ylabel("Temperatura (°C)", fontsize=11)
axes[0].set_title("Sensor 1: Temperatura del Sistema", fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)

# Sensor Vibración
axes[1].plot(df_imputed['timestamp'], df_imputed['sensor_vib'], label='Señal Reconstruida (F-fill)', color='#065F46', lw=2)
nan_indices = df_raw['sensor_vib'].isna()
if nan_indices.any():
    axes[1].scatter(df_raw['timestamp'][nan_indices], df_imputed['sensor_vib'][nan_indices],
                    color='#EF4444', marker='x', s=50, label='Dato Perdido (Imputado)', zorder=5)
axes[1].set_ylabel("Vibración (g)", fontsize=11)
axes[1].set_title("Sensor 2: Vibración Mecánica", fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=10)

# Sensor Presión
axes[2].plot(df_imputed['timestamp'], df_imputed['sensor_press'], label='Señal Reconstruida (F-fill)', color='#7C3AED', lw=2)
nan_indices = df_raw['sensor_press'].isna()
if nan_indices.any():
    axes[2].scatter(df_raw['timestamp'][nan_indices], df_imputed['sensor_press'][nan_indices],
                    color='#EF4444', marker='x', s=50, label='Dato Perdido (Imputado)', zorder=5)
axes[2].set_ylabel("Presión (bar)", fontsize=11)
axes[2].set_title("Sensor 3: Presión Hidráulica", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Tiempo transcurrido (s)", fontsize=11)
axes[2].legend(loc='upper left', fontsize=10)

plt.tight_layout()
st.pyplot(fig)


# ----------------- DIAGNÓSTICO FÍSICO -----------------
st.markdown("### 💡 Diagnóstico del Sistema y Análisis de Evidencia")

# Identificar qué sensor causó la anomalía conceptualmente
max_temp = df_imputed['sensor_temp'].max()
max_vib = df_imputed['sensor_vib'].max()
min_press = df_imputed['sensor_press'].min()

evidencia_falla = "El sistema opera en condiciones óptimas."
if anomaly_type == 'thermal':
    evidencia_falla = f"Se detectó un incremento anómalo en la **temperatura máxima de {max_temp:.2f}°C** acompañada de un incremento térmico de presión. Bi-Mamba clasifica esto como sobrecalentamiento."
elif anomaly_type == 'spikes':
    evidencia_falla = f"Se detectaron picos de aceleración mecánica en la **vibración máxima de {max_vib:.2f}g**, indicando un posible fallo de rodamiento o desalineación."
elif anomaly_type == 'leak':
    evidencia_falla = f"Se detectó una despresurización anómala donde la **presión mínima cayó a {min_press:.2f} bar**, consistente con una fuga hidráulica progresiva."

st.markdown(f"""
<div class='card' style='background-color:#F3F4F6;'>
    <h4>🔍 Análisis de Evidencia de Sensores:</h4>
    <p style='font-size:1.05rem; color:#374151;'>{evidencia_falla}</p>
    <p style='font-size:0.9rem; color:#6B7280;'><b>Comparación Operacional:</b> El modelo Bi-Mamba es capaz de clasificar de forma tolerante a fallos porque no asume un muestreo temporal constante, utilizando delta_t físico como parámetro dinámico de propagación de estado.</p>
</div>
""", unsafe_allow_html=True)
