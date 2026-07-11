# Informe Técnico: Bi-Mamba para Detección Tolerante a Fallos en Sensores IIoT con Secuencias Irregulares

Este informe técnico documenta el diseño, la formulación matemática, la arquitectura de red y el análisis experimental para el **Proyecto 3** del Examen Final del curso **CC0C2: Procesamiento de Lenguaje Natural**.

---

## 1. Introducción y Planteamiento del Problema

En entornos de Manufactura Avanzada e Industrial Internet of Things (IIoT), la telemetría continua de maquinaria rotativa o hidráulica es crítica para el mantenimiento predictivo. No obstante, las transmisiones inalámbricas en plantas reales enfrentan dos retos físicos persistentes:
1.  **Muestreo Temporal Irregular ($\Delta t$ variable):** Debido a la latencia de red, colisiones en canales inalámbricos o muestreo adaptativo de baja potencia, la diferencia de tiempo entre lecturas consecutivas no es constante.
2.  **Pérdida de Datos (Paquetes Perdidos / NaNs):** Las interferencias electromagnéticas del entorno industrial provocan dropouts donde algunos de los transductores (temperatura, vibración o presión) no transmiten su lectura temporalmente.

Las redes neuronales recurrentes clásicas (como LSTM y GRU) asumen que la secuencia de datos tiene un intervalo de muestreo constante ($\Delta t$ implícito igual a $1.0$). Cuando se enfrentan a distorsiones temporales físicas del reloj y a valores imputados debido a pérdidas de red, su capacidad de modelar correctamente la dinámica física del sistema se degrada drásticamente.

### Propuesta del Proyecto
Para solucionar esto, se propone un clasificador basado en **Bi-Mamba** (State Space Model Bidireccional). A diferencia de las redes recurrentes, Mamba incorpora explícitamente la dimensión del tiempo físico real ($\Delta t$) dentro de su ecuación diferencial de estado mediante una discretización continua adaptativa.

---

## 2. Formulación Matemática de Mamba

Mamba es una arquitectura basada en **Modelos de Espacio de Estados Estructurados (S4)**. Mapea una señal de entrada unidimensional continua $x(t) \in \mathbb{R}$ a un estado latente continuo $h(t) \in \mathbb{R}^N$ y produce una salida continua $y(t) \in \mathbb{R}$ a través de un sistema de ecuaciones diferenciales lineales:

$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t)$$

Donde:
*   $A \in \mathbb{R}^{N \times N}$ es la matriz de transición del sistema latente.
*   $B \in \mathbb{R}^{N \times 1}$ es la matriz de proyección de la señal de entrada.
*   $C \in \mathbb{R}^{1 \times N}$ es la matriz de proyección de salida.

### A. Discretización Adaptativa mediante Tiempo Físico $\Delta t$
Para procesar señales digitales discretas en un computador, el sistema continuo debe transformarse en una formulación discreta. En lugar de asumir un paso constante, Mamba calcula dinámicamente un tamaño de paso $\Delta_t$ en cada instante $t$, combinando el segundero físico real del sensor y la selectividad aprendida por la red.

En el código de [app/model.py](file:///Users/cesar/Desktop/NLP%20EF/app/model.py), esto se computa forzando la positividad del tiempo real con una función **Softplus** y proyectando el paso aprendido:

$$\Delta_t = \text{Sigmoid}(W_{\text{proj}} \cdot x_t) \times \text{Softplus}(\Delta t_{\text{físico}})$$

Una vez calculado $\Delta_t$, el sistema se discretiza usando la regla de integración exponencial para $A$ y la regla del trapecio modificada para $B$:

$$\overline{A}_t = \exp(\Delta_t \cdot A)$$
$$\overline{B}_t = \Delta_t \cdot B_t$$

### B. Ecuación de Recurrencia Selectiva
Con las matrices discretizadas en el tiempo, el estado latente $h_t$ se actualiza recurrentemente en cada paso $t$:

$$h_t = \overline{A}_t h_{t-1} + \overline{B}_t x_t$$
$$y_t = C_t h_t$$

### C. Estabilización de la Matriz $A$
Para garantizar que los estados ocultos $h_t$ no diverjan numéricamente a valores infinitos (inestabilidad de Lyapunov), los valores de la diagonal de la matriz $A$ se restringen estrictamente a valores negativos reales:

$$A = -\exp(A_{\text{param}})$$

Al forzar que los elementos de $A$ sean negativos, el término de decaimiento temporal $\overline{A}_t = \exp(\Delta_t \cdot A)$ siempre estará acotado en el intervalo $[0.0, 1.0)$, garantizando la estabilidad asintótica del modelo.

---

## 3. Arquitectura del Sistema

El pipeline del proyecto está compuesto por los siguientes módulos integrados en la carpeta [app/](file:///Users/cesar/Desktop/NLP%20EF/app):

1.  **Imputación Robusta ([app/utils.py](file:///Users/cesar/Desktop/NLP%20EF/app/utils.py)):**
    *   Implementa *Forward-Fill* como la técnica de imputación prioritaria. Si un sensor pierde datos, propaga su última lectura válida conocida bajo el supuesto físico de inercia del sensor. En caso de fallas de inicio absoluto, aplica *Backward-Fill*.
2.  **Normalización:**
    *   Ajusta un `StandardScaler` con los datos de entrenamiento históricos y lo guarda en disco para centrar todas las magnitudes de sensores en escala homogénea (media 0, varianza 1).
3.  **Arquitectura Bidireccional (Bi-Mamba):**
    *   **Rama Forward:** Procesa la secuencia de manera cronológica (pasado a futuro).
    *   **Rama Backward:** Invierte el orden temporal de la secuencia y de los deltas de tiempo utilizando `torch.flip(out, dims=[1])`, la procesa con un bloque SSM y re-invierte el resultado.
    *   **Fusión:** Ambas ramas se concatenan (`2 * d_model`), se promedian temporalmente (*Global Average Pooling*) para retener anomalías transitorias tempranas y se clasifican mediante una capa sigmoide.

---

## 4. Diseño Experimental y Resultados

El experimento de comparación se diseñó bajo condiciones controladas de reproducibilidad (semilla aleatoria fijada en `42`), entrenando tres clasificadores secuenciales con el mismo tamaño de capacidad interna (`hidden_dim = 32`) sobre 200 secuencias de sensores IIoT de longitud 500.

### Resultados de Métricas en Validación (Split 70/30)

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM Baseline** | 93.33% | 96.43% | 90.00% | 93.10% |
| **GRU Baseline** | 95.00% | 100.00% | 90.00% | 94.74% |
| **Bi-Mamba (Propuesto)** | **96.67%** | **100.00%** | **93.33%** | **96.55%** |

### Análisis del Rendimiento
*   **Bi-Mamba** superó a ambos baselines, alcanzando el F1-Score más alto (**96.55%**). Esto se debe a su capacidad de interpretar físicamente el espaciamiento temporal irregular mediante la inyección del parámetro $\Delta t$, logrando una clasificación tolerante a fallas y pérdidas de comunicación.
*   **Análisis de Latencia (CPU):** Durante las pruebas en CPU local, la LSTM registró una velocidad ligeramente superior a la GRU y a Mamba. Esto ocurre porque PyTorch cuenta con implementaciones internas en C++ y llamadas vectoriales (GEMM) altamente especializadas para el kernel de la LSTM. Mamba, al implementarse en código secuencial de bucle puro de Python para CPU en este proyecto, tiene una sobrecarga de latencia que puede resolverse compilando los kernels en CUDA para su ejecución en GPU.

---

## 5. Limitaciones del Sistema

A pesar de su alto rendimiento predictivo, el sistema presenta limitaciones técnicas que deben considerarse para despliegues a gran escala:

1.  **Ejecución en CPU no vectorizada:**
    El bucle recursivo de Mamba (`for t in range(seq_len)`) está implementado nativamente en PyTorch de forma secuencial en Python. En entornos de GPU, Mamba requiere un kernel personalizado en CUDA para paralelizar esta operación a través del algoritmo de escaneo asociativo (*Associative Scan*). En CPU, la ejecución secuencial limita el rendimiento de rendimiento de datos (*throughput*).
2.  **Sensibilidad a la Calidad de la Imputación Inicial:**
    Si ocurre una pérdida de datos extremadamente larga al principio de una secuencia (por ejemplo, el 100% de los primeros 100 pasos son nulos), el algoritmo recae en los valores por defecto estáticos (`defaults`), lo que puede sesgar las predicciones iniciales del modelo.
3.  **Vulnerabilidad a Intervalos del Reloj Anómalos:**
    Si la tasa temporal irregular $\Delta t$ experimenta picos de retraso excesivos (por ejemplo, interrupción de comunicación por varias horas), la discretización matemática del estado $\overline{A}_t = \exp(\Delta_t \cdot A)$ decae muy cerca de cero, provocando un "olvido" parcial del estado histórico acumulado por la red.

---

## 6. Conclusiones

*   La inyección directa del paso de tiempo físico $\Delta t$ en la discretización de la capa del espacio de estados de Mamba provee una representación dinámica del sistema superior a las redes recurrentes tradicionales que omiten el espaciamiento temporal real.
*   La naturaleza bidireccional de Bi-Mamba permite recolectar evidencias de fallas transitorias o graduales tanto del futuro como del pasado, estabilizando la predicción de fallas tolerante al ruido de medición.
*   Para despliegue en tiempo real en plantas de manufactura, el pipeline es robusto, ligero y fácil de empaquetar en Docker, proporcionando métricas de latencia estables para la toma de decisiones en el cockpit de Streamlit.
