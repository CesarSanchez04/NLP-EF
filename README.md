# Bi-Mamba para Detección Tolerante a Fallos en Sensores IIoT con Secuencias Irregulares

Este repositorio contiene el desarrollo progresivo para el **Proyecto 3** del Examen Final del curso **CC0C2: Procesamiento de Lenguaje Natural**. 

El sistema implementa un clasificador de series temporales de sensores industriales mediante una arquitectura de **Mamba Bidireccional (Bi-Mamba)** construida nativamente en PyTorch, permitiendo capturar el contexto temporal hacia adelante y hacia atrás, y adaptándose a intervalos de muestreo irregulares ($\Delta t$).

---

## 🚧 Estado del Desarrollo (Construcción Progresiva)

De acuerdo con las directrices de evaluación del examen final para asegurar el historial de autoría técnica y progreso regular, el proyecto se está construyendo paso a paso:

* **Día 1: Simulación y Creación del Dataset (Completado):** 
  Se ha desarrollado el cuaderno de simulación [creacion_dataset.ipynb](file:///Users/cesar/Desktop/NLP%20EF/notebooks/creacion_dataset.ipynb) en la carpeta `notebooks/`. Este cuaderno modela el comportamiento físico de tres sensores (Temperatura, Vibración y Presión), inyecta marcas de tiempo irregulares ($\Delta t$), simula pérdidas de paquetes (NaNs) y valida visualmente tres métodos de imputación (Zero, Mean, Forward-fill).
* **Día 2: Exportar Utilidades y Entrenamiento de Líneas Base (En progreso):** 
  Fase dedicada a modularizar el generador de datos en `utils.py` e implementar y entrenar las redes recurrentes tradicionales (LSTM y GRU) como benchmark.

---

## 🚀 Ejecución y Reproducibilidad con Docker

El proyecto está contenerizado para garantizar que se ejecute en idénticas condiciones (CPU/GPU) en cualquier sistema operativo.

### Prerrequisitos
- Tener instalado [Docker](https://www.docker.com/products/docker-desktop/) y Docker Compose.

### Instrucciones para levantar los servicios:

1. **Construir e iniciar los contenedores:**
   En la raíz del proyecto, ejecuta:
   ```bash
   docker compose up --build
   ```

2. **Acceder a los entornos de trabajo expuestos:**
   - **Servidor de Jupyter Notebook (Puerto 8888):** [http://localhost:8888](http://localhost:8888)  
     *(Para abrir y ejecutar de manera interactiva el cuaderno de creación de dataset).*
   - **Interfaz de Usuario Streamlit (Puerto 8501):** [http://localhost:8501](http://localhost:8501)  
     *(Servidor base de visualización, la demo de inferencia real se integrará en el Día 4).*

3. **Detener la aplicación:**
   Para apagar los contenedores, ejecuta:
   ```bash
   docker compose down
   ```

---

## 📂 Estructura del Repositorio

* **`notebooks/`**: Cuadernos interactivos de desarrollo de modelos y datos.
  * [creacion_dataset.ipynb](file:///Users/cesar/Desktop/NLP%20EF/notebooks/creacion_dataset.ipynb): Simulación del dataset IIoT y métodos de imputación.
* **`app/`**: Directorio principal de código fuente (en desarrollo incremental).
  * [main.py](file:///Users/cesar/Desktop/NLP%20EF/app/main.py): Interfaz base de Streamlit.
  * `model.py` y `utils.py` (archivos vacíos listos para ser implementados modularmente).
* **`data/`**: Carpeta destinada a guardar los conjuntos de datos en formato CSV.
* **`results/`**: Carpeta destinada a guardar las métricas de rendimiento y curvas de robustez.
* **`docs/`**: Carpeta destinada a guardar la documentación técnica detallada.
