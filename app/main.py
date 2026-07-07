import streamlit as st

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
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Contenido principal
st.markdown("<h1 class='main-title'>🏭 Plataforma Bi-Mamba para Telemetría IIoT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detección Tolerante a Fallos en Sensores con Muestreo Temporal Irregular y Pérdida de Datos</p>", unsafe_allow_html=True)

st.markdown("""
<div class='card'>
    <h3>🚧 Fase de Desarrollo Activa</h3>
    <p>El proyecto se está construyendo de forma incremental paso a paso:</p>
    <ul>
        <li><strong>Día 1 (Completado):</strong> Simulación de dataset y tratamiento de NaNs en el cuaderno.</li>
        <li><strong>Día 2 (En progreso):</strong> Exportar utilidades de datos y entrenamiento de líneas base (LSTM/GRU).</li>
        <li><strong>Día 3 (Siguiente):</strong> Integración de arquitectura Bi-Mamba (Selective Scan con delta t físico).</li>
        <li><strong>Día 4:</strong> Integración final de la inferencia en tiempo real en esta interfaz.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.info("El servidor Streamlit se ha iniciado correctamente. La interfaz interactiva de inferencia se integrará en el Día 4.")
