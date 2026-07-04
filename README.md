# Bi-Mamba para Detección Tolerante a Fallos en Sensores IIoT con Secuencias Irregulares

Este repositorio contiene la implementación y el despliegue del **Proyecto 3** para el Examen Final del curso **CC0C2: Procesamiento de Lenguaje Natural**. 

El sistema implementa un clasificador de series temporales de sensores industriales mediante una arquitectura de **Mamba Bidireccional (Bi-Mamba)** construida nativamente en PyTorch, permitiendo capturar el contexto temporal hacia adelante y hacia atrás, y adaptándose a intervalos de muestreo irregulares ($\Delta t$).

---

## 🚀 Despliegue con Docker (Recomendado y Replicable)

Para garantizar la reproducibilidad completa sin importar el sistema operativo o hardware (CPU/GPU), se proporciona una configuración de Docker.

### Prerrequisitos
- Tener instalado [Docker](https://www.docker.com/products/docker-desktop/) y Docker Compose en tu máquina.

### Pasos para Ejecutar la Aplicación:

1. **Construir y levantar el contenedor:**
   En la raíz del proyecto, ejecuta el siguiente comando en tu terminal:
   ```bash
   docker compose up --build
   ```

2. **Acceder a la Interfaz de Usuario:**
   Una vez que el contenedor esté corriendo, abre tu navegador web e ingresa a:
   ```
   http://localhost:8501
   ```

3. **Detener la aplicación:**
   Para apagar el contenedor, presiona `Ctrl + C` en la terminal o ejecuta:
   ```bash
   docker compose down
   ```

---

## 🛠️ Ejecución Local (Sin Docker)

Si prefieres ejecutar el proyecto directamente en tu entorno local mediante un entorno virtual de Python:

1. **Crear y activar un entorno virtual:**
   ```bash
   # En macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate

   # En Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la interfaz de Streamlit:**
   ```bash
   streamlit run app/main.py
   ```

---

## 📂 Estructura del Repositorio

De acuerdo con las pautas del examen final, el proyecto se organiza de la siguiente manera:

* **`app/`**: Código fuente de la aplicación de despliegue.
  * [model.py](file:///Users/cesar/Desktop/NLP%20EF/app/model.py): Definición matemática de los bloques Selective SSM (S6), BiMambaBlock y el detector clasificador.
  * [utils.py](file:///Users/cesar/Desktop/NLP%20EF/app/utils.py): Simulación de datos de sensores IIoT, adición de ruido/NaNs y métodos de imputación.
  * [main.py](file:///Users/cesar/Desktop/NLP%20EF/app/main.py): Interfaz visual de usuario con Streamlit.
* **`data/`**: Contiene los conjuntos de datos o instrucciones para su obtención/generación.
* **`results/`**: Carpeta destinada a guardar las métricas de evaluación offline, reportes y gráficas comparativas.
* **`docs/`**: Documentación técnica, explicaciones arquitectónicas del modelo Bi-Mamba y limitaciones detectadas.
* **`Dockerfile`**: Configuración del contenedor de la aplicación.
* **`docker-compose.yml`**: Orquestación y mapeo de puertos del contenedor.
* **`requirements.txt`**: Lista completa de librerías Python y versiones utilizadas.

---

## 🧠 Aspectos Clave de la Arquitectura

### 1. Manejo de la Irregularidad Temporal ($\Delta t$)
En entornos industriales (IIoT), los sensores a menudo registran lecturas a intervalos asíncronos debido a latencias de red o suspensión por ahorro de energía. Bi-Mamba soluciona esto integrando el tiempo real observado $\Delta t_{\text{físico}}$ en la discretización de la matriz de transición del estado oculto ($A$):

$$\Delta_t = \text{Softplus}(\text{Linear}(x_t) + \log(\Delta t_{\text{físico}}))$$

### 2. Bidireccionalidad (Bi-Mamba)
El modelo procesa la señal en dos direcciones simultáneas:
* **Forward (Hacia adelante):** Para modelar el comportamiento acumulativo y degradación temporal del sensor.
* **Backward (Hacia atrás):** Para capturar cambios y patrones de retorno que permiten un pre-diagnóstico más robusto.

Ambas salidas se fusionan linealmente antes de la clasificación final de fallas.

### 3. Tratamiento de Pérdida de Datos (NaNs)
El módulo incluye baselines de imputación temporal (Forward-fill, Imputación por media) que demuestran cómo tratar la pérdida de datos antes de enviarlos a la red neuronal de espacio de estados.
