# Bi-Mamba para Detección Tolerante a Fallos en Sensores IIoT con Secuencias Irregulares

Este repositorio contiene el desarrollo completo para el **Proyecto 3** del Examen Final del curso **CC0C2: Procesamiento de Lenguaje Natural**. 

El sistema implementa un clasificador de series temporales de sensores industriales mediante una arquitectura de **Mamba Bidireccional (Bi-Mamba)** construida nativamente en PyTorch, permitiendo capturar el contexto temporal hacia adelante y hacia atrás, y adaptándose a intervalos de muestreo irregulares ($\Delta t$).

---

## Guía de Reproducción de Experimentos

El proyecto puede ser reproducido de dos formas: localmente o utilizando contenedores de Docker.

### Opción A: Ejecución Local en Terminal
Para reproducir los experimentos y levantar el panel interactivo en tu máquina, sigue este orden de comandos:

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generar el Dataset:**
    Genera el archivo `data/sensores_iiot_simulados.csv` simulando telemetría irregular, ruido y pérdida de datos:
    ```bash
    python generate_dataset.py
    ```
3.  **Entrenar los Modelos:**
    Entrena las líneas base (LSTM y GRU) y la propuesta (Bi-Mamba), exportando los pesos (.pth) y el escalador (.pkl) a la carpeta `results/`:
    ```bash
    python train_models.py
    ```
4.  **Lanzar la Aplicación Streamlit:**
    Inicia la interfaz de usuario interactiva para realizar simulaciones en tiempo real:
    ```bash
    streamlit run app/main.py
    ```

### Opción B: Contenedorización con Docker
El proyecto está contenerizado para garantizar que se ejecute en idénticas condiciones en cualquier sistema operativo.

1.  **Construir y levantar contenedores:**
    ```bash
    docker compose up --build
    ```
2.  **Acceder a los servicios expuestos:**
    *   **Interfaz de Usuario Streamlit:** [http://localhost:8501](http://localhost:8501) (Monitoreo en tiempo real de telemetría y diagnóstico físico).
    *   **Servidor Jupyter Notebook:** [http://localhost:8888](http://localhost:8888) (Entorno experimental interactivo).
3.  **Apagar los contenedores:**
    ```bash
    docker compose down
    ```

---

## Resultados Experimentales

Evaluación comparativa de los modelos entrenados con semilla fija `42` en el conjunto de validación (30% del dataset):

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM Baseline** | 93.33% | 96.43% | 90.00% | 93.10% |
| **GRU Baseline** | 95.00% | 100.00% | 90.00% | 94.74% |
| **Bi-Mamba (Propuesto)** | **96.67%** | **100.00%** | **93.33%** | **96.55%** |

### Conclusión Técnica
La variante propuesta **Bi-Mamba** superó a ambos baselines, alcanzando el F1-Score más alto (**96.55%**), demostrando una alta tolerancia ante pérdidas dinámicas de datos e irregularidades en el reloj de muestreo gracias a la inyección física del parámetro $\Delta t$ dentro del proceso de discretización de estados.

---

## Evidencia y Defensa Técnica
*   **Video de Exposición del Proyecto (más de 10 minutos):** [Enlace al Video de Defensa](COLOQUE_EL_ENLACE_AQUI) *(Reemplace este marcador por el enlace final de YouTube/Drive)*
*   **Explicación Teórica Detallada:** Consulta el archivo de documentación técnica [docs/informe_tecnico.md](file:///Users/cesar/Desktop/NLP%20EF/docs/informe_tecnico.md) para comprender la formulación matemática de la discretización del espacio de estados y análisis de limitaciones.

---

## Estructura del Repositorio

*   [generate_dataset.py](file:///Users/cesar/Desktop/NLP%20EF/generate_dataset.py): Script de simulación física del dataset IIoT.
*   [train_models.py](file:///Users/cesar/Desktop/NLP%20EF/train_models.py): Script de entrenamiento comparativo y guardado de resultados.
*   **`app/`**: Código fuente modularizado del sistema.
    *   [app/model.py](file:///Users/cesar/Desktop/NLP%20EF/app/model.py): Definición de redes neuronales (LSTM, GRU, Mamba y Bi-Mamba).
    *   [app/utils.py](file:///Users/cesar/Desktop/NLP%20EF/app/utils.py): Funciones de imputación, normalización y carga en tensores de PyTorch.
    *   [app/main.py](file:///Users/cesar/Desktop/NLP%20EF/app/main.py): Aplicación interactiva de Streamlit.
*   **`notebooks/`**: Cuadernos interactivos.
    *   [notebooks/proyecto_final.ipynb](file:///Users/cesar/Desktop/NLP%20EF/notebooks/proyecto_final.ipynb): Sandbox experimental del proyecto.
*   **`results/`**: Almacén de checkpoints de modelos (`.pth`), escalador (`.pkl`), curvas de aprendizaje y matrices de confusión.
*   **`docs/`**: Documentación teórica del Examen Final.
    *   [docs/informe_tecnico.md](file:///Users/cesar/Desktop/NLP%20EF/docs/informe_tecnico.md): Informe técnico de arquitectura, formulación matemática y limitaciones.
