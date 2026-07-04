### Examen Final CC0C2: Procesamiento de Lenguaje Natural

#### Condiciones de entrega

La fecha de entrega del examen final será el **16 de julio**.

Cada estudiante deberá presentar obligatoriamente un  su **repositorio  los resultados del proyecto**. No se aceptará únicamente un archivo comprimido, capturas de pantalla o un cuaderno aislado sin trazabilidad.

El repositorio debe contener como mínimo:

1. `README.md` con descripción del problema, instalación, ejecución y reproducción.
2. Cuaderno técnico en formato `.ipynb`.
3. Código fuente organizado.
4. Carpeta `data/` o instrucciones claras para obtener los datos.
5. Carpeta `results/` con métricas, tablas, salidas, errores y evidencias.
6. Carpeta `docs/` con explicación técnica, arquitectura y limitaciones.
7. Archivo de dependencias, por ejemplo `requirements.txt`, `environment.yml` o `pyproject.toml`.
8. Evidencia del despliegue, por ejemplo URL, capturas, instrucciones o archivo Docker.
9. Enlace al video de más de 10 minutos.
10. Registro claro de cómo reproducir los experimentos.

El repositorio debe mostrar un proceso de trabajo progresivo. No se evaluará de la misma forma un proyecto desarrollado de manera continua que un proyecto subido completo al final sin historial verificable.

#### Control de progreso mediante commits

Para evitar trabajos hechos a última hora, se revisará el historial del repositorio.

El repositorio debe mostrar commits distribuidos en el tiempo, con avances reales y mensajes comprensibles. Se espera observar, como mínimo:

1. Commit inicial con estructura del proyecto.
2. Commit de preparación de datos o corpus.
3. Commit de línea base.
4. Commit de modelo, pipeline o arquitectura principal.
5. Commit de evaluación y resultados.
6. Commit de despliegue.
7. Commit final de documentación.

No se considerarán commits sustantivos aquellos que solo cambien espacios, títulos, formato, nombres de archivos o texto superficial sin avance técnico.

#### Penalización por trabajo de última hora

La nota final podrá recibir descuentos cuando el historial del repositorio muestre trabajo concentrado artificialmente al final.

Criterios de descuento:

1. **Sin descuento**: commits distribuidos en varios días, con avances técnicos verificables.
2. **Hasta -0.5 puntos**: historial débil, pocos commits sustantivos o muchos cambios menores.
3. **Hasta -1 punto**: la mayoría del trabajo técnico aparece concentrado en las últimas 48 horas.
4. **Hasta -2 puntos**: repositorio creado o llenado casi por completo el último día.
5. **Hasta -3 puntos**: historial artificial, un único commit masivo, código pegado sin evolución, o evidencia fuerte de trabajo generado sin comprensión.

El descuento se aplicará sobre la nota final del examen. El objetivo no es castigar el uso de Git, sino exigir evidencia mínima de proceso, iteración y autoría técnica.

#### Restricción contra proyectos demasiado generales

Cada proyecto debe presentar obligatoriamente:

1. Problema específico.
2. Dataset, corpus o conjunto de entradas.
3. Línea base.
4. Variante propuesta.
5. Métricas.
6. Análisis de errores.
7. Despliegue mínimo.
8. Repositorio reproducible.
9. Discusión de limitaciones.
10. Relación explícita con los cuadernos del curso.

No se aceptarán proyectos formulados de manera genérica como:

1. "Haré un chatbot".
2. "Usaré RAG".
3. "Haré un sistema con IA".
4. "Usaré LangChain".
5. "Usaré un modelo multimodal".
6. "Haré una app con LLM".

El proyecto debe demostrar una pregunta técnica evaluable.

Ejemplo aceptable:

"Comparar un sistema RAG con recuperación BM25, recuperación densa y recuperación híbrida para responder preguntas sobre un corpus jurídico, midiendo recall@k, exactitud de respuesta, fidelidad al contexto y errores de citación."

Ejemplo no aceptable:

"Sistema RAG para documentos jurídicos."


### Temas de proyecto con alcance cerrado

#### Proyecto 1. Sistema multiagente para generación automática de consultas DAX en Power BI

El estudiante deberá construir un sistema que convierta preguntas en lenguaje natural en consultas DAX para un modelo semántico de Power BI o un esquema tabular equivalente.

Alcance mínimo:

1. Al menos 5 tablas o entidades del modelo.
2. Al menos 20 preguntas en lenguaje natural.
3. Al menos 20 consultas DAX esperadas o criterios de validación.
4. Agente generador.
5. Agente evaluador.
6. Orquestación iterativa.
7. Validación sintáctica o semántica.
8. Comparación contra una línea base sin agente evaluador.

Métricas sugeridas:

1. Porcentaje de consultas sintácticamente válidas.
2. Porcentaje de consultas ejecutables.
3. Coincidencia semántica con la respuesta esperada.
4. Número promedio de iteraciones de refinamiento.
5. Errores por tipo.

Despliegue mínimo:

1. API o interfaz donde se ingrese una pregunta.
2. Salida con consulta DAX generada.
3. Resultado de validación.
4. Explicación breve del razonamiento del sistema.

El estudiante debe explicar la diferencia entre generar una consulta aparentemente correcta y generar una consulta válida respecto al esquema.



#### Proyecto 2. Detección automática de contradicciones normativas en textos jurídicos peruanos mediante RAG e inferencia de lenguaje natural

El estudiante deberá construir un sistema que recupere fragmentos normativos y clasifique si existe contradicción, compatibilidad, especificación, excepción o ausencia de relación.

Alcance mínimo:

1. Corpus jurídico documentado.
2. Al menos 30 artículos o fragmentos normativos.
3. Al menos 20 pares de textos para evaluación.
4. Recuperación semántica, léxica o híbrida.
5. Clasificación de relación normativa.
6. Explicación con evidencia textual.
7. Comparación contra una línea base simple.

Métricas sugeridas:

1. Recall@k de recuperación.
2. Precision@k.
3. Exactitud o F1 de clasificación.
4. Porcentaje de respuestas con cita correcta.
5. Análisis de falsos positivos y falsos negativos.

Despliegue mínimo:

1. Entrada con artículo, consulta o fragmento jurídico.
2. Recuperación de normas relacionadas.
3. Clasificación de la relación.
4. Explicación fundamentada con citas.

No será suficiente decir que dos normas se contradicen. El estudiante debe justificar la contradicción usando evidencia recuperada.



#### Proyecto 3. Bi-Mamba para detección tolerante a fallos en sensores IIoT con secuencias irregulares

Este proyecto se aceptará solo si el estudiante lo conecta explícitamente con modelos secuenciales, evaluación experimental, arquitectura de aprendizaje profundo y despliegue. No debe convertirse en un proyecto puramente electrónico o industrial sin relación con los temas computacionales del curso.

Alcance mínimo:

1. Dataset de sensores real, público o sintético documentado.
2. Simulación o presencia de irregularidad temporal.
3. Tratamiento de datos faltantes o ruido.
4. Línea base, por ejemplo LSTM, GRU, Transformer pequeño o modelo estadístico.
5. Variante propuesta basada en Mamba, Bi-Mamba o modelo secuencial equivalente.
6. Evaluación comparativa.
7. Análisis de fallos.

Métricas sugeridas:

1. Accuracy, precision, recall y F1.
2. AUC, si corresponde.
3. Latencia de inferencia.
4. Robustez ante datos faltantes.
5. Rendimiento ante ruido.

Despliegue mínimo:

1. Servicio que reciba una secuencia.
2. Predicción de estado normal o falla.
3. Explicación de la salida.
4. Registro de métricas básicas.

El estudiante debe explicar qué aporta el modelo secuencial frente a una línea base simple.



#### Proyecto 4. Predicción de elegibilidad de postulantes en ofertas laborales mediante NLP

El estudiante deberá construir un sistema que compare postulantes y ofertas laborales usando procesamiento de lenguaje natural.

Alcance mínimo:

1. Al menos 50 ofertas laborales o descripciones de puestos.
2. Al menos 50 perfiles o CV sintéticos, anonimizados o documentados.
3. Representación de habilidades, experiencia o requisitos.
4. Línea base con TF-IDF, BM25 o similitud simple.
5. Variante con embeddings o modelo de lenguaje.
6. Evaluación de ranking o clasificación.
7. Discusión de sesgo y privacidad.

Métricas sugeridas:

1. Accuracy, precision, recall y F1.
2. nDCG o MRR para ranking.
3. Comparación entre línea base y variante.
4. Análisis de casos mal clasificados.
5. Revisión de sesgos potenciales.

Despliegue mínimo:

1. Entrada con una oferta laboral y un perfil.
2. Salida con elegibilidad o puntaje de compatibilidad.
3. Explicación textual basada en evidencia.
4. Advertencia sobre limitaciones y uso responsable.

No será suficiente devolver un porcentaje. El estudiante debe explicar qué evidencia textual sostiene el resultado.



#### Proyecto 5. Algoritmos competitivos para recuperación y evaluación en sistemas RAG

El estudiante deberá comparar algoritmos de recuperación de información y analizar su impacto en un sistema RAG.

Alcance mínimo:

1. Corpus con al menos 100 documentos o fragmentos.
2. Al menos 30 consultas de evaluación.
3. Línea base con BM25 o TF-IDF.
4. Variante con embeddings densos.
5. Variante híbrida o reranking.
6. Métricas de recuperación.
7. Análisis de costo, latencia y calidad.

Métricas sugeridas:

1. Precision@k.
2. Recall@k.
3. MRR.
4. nDCG.
5. Latencia promedio.
6. Memoria aproximada.
7. Comparación de errores por tipo de consulta.

Despliegue mínimo:

1. Buscador donde se ingrese una consulta.
2. Retorno de top-k documentos.
3. Comparación visible entre al menos dos métodos.
4. Métricas o explicación del ranking.

No será suficiente usar una biblioteca. El estudiante debe explicar el algoritmo, el índice, el costo y el efecto sobre la recuperación.



#### Proyecto 6. Evaluador automático de sistemas RAG con métricas, citas y detección de alucinaciones

El estudiante deberá construir un evaluador que determine si las respuestas de un sistema RAG están fundamentadas en el contexto recuperado.

Alcance mínimo:

1. Corpus documental.
2. Al menos 50 preguntas.
3. Dos variantes de RAG o una línea base y una variante mejorada.
4. Evaluación de contexto recuperado.
5. Evaluación de respuesta generada.
6. Verificación de citas.
7. Reporte de errores.

Métricas sugeridas:

1. Context precision.
2. Context recall.
3. Answer relevance.
4. Faithfulness o groundedness.
5. Porcentaje de citas correctas.
6. Porcentaje de respuestas no sustentadas.
7. Errores del recuperador frente a errores del generador.

Despliegue mínimo:

1. Entrada con pregunta y respuesta.
2. Contexto recuperado.
3. Evaluación automática.
4. Reporte de si la respuesta está sustentada o no.

Este proyecto debe priorizar evaluación, no solo generación.



#### Proyecto 7. Sistema multimodal de búsqueda y respuesta sobre imágenes y texto

El estudiante deberá construir un sistema de recuperación multimodal usando imágenes y texto.

Alcance mínimo:

1. Al menos 100 imágenes o pares texto-imagen.
2. Al menos 30 consultas textuales.
3. Baseline usando captions o metadatos textuales.
4. Variante usando embeddings multimodales.
5. Evaluación de recuperación.
6. Análisis de errores visuales.
7. Despliegue mínimo.

Métricas sugeridas:

1. Recall@k.
2. Precision@k.
3. Similaridad promedio.
4. Casos de recuperación incorrecta.
5. Comparación entre texto plano y embedding multimodal.

Despliegue mínimo:

1. Entrada con consulta textual.
2. Retorno de imágenes relevantes.
3. Explicación de la similitud.
4. Visualización de resultados.

No será suficiente mostrar imágenes. El estudiante debe explicar la representación conjunta texto-imagen.



#### Proyecto 8. Plataforma mínima de LLMOps y PromptOps para control de prompts, versiones y regresiones

El estudiante deberá construir una plataforma pequeña para registrar, comparar y evaluar versiones de prompts o configuraciones de modelos.

Alcance mínimo:

1. Al menos 3 versiones de prompts.
2. Al menos 20 casos de prueba.
3. Registro de modelo, parámetros y versión.
4. Evaluación automática de salidas.
5. Comparación entre versiones.
6. Detección de regresiones.
7. Reporte reproducible.

Métricas sugeridas:

1. Tasa de respuestas correctas.
2. Cumplimiento de formato.
3. Consistencia.
4. Latencia.
5. Costo aproximado, si corresponde.
6. Número de regresiones detectadas.

Despliegue mínimo:

1. Interfaz o API para ejecutar pruebas.
2. Comparación entre versiones de prompt.
3. Reporte de mejora o regresión.
4. Historial de resultados.

No será suficiente mostrar una interfaz. El estudiante debe explicar cómo se controla la calidad cuando cambia el prompt.



#### Proyecto 9. Despliegue MLOps de un sistema NLP con API, monitoreo y evaluación continua

El estudiante deberá convertir un modelo o pipeline NLP en un servicio reproducible.

Alcance mínimo:

1. Tarea NLP concreta, por ejemplo clasificación, NER, similitud textual, análisis de sentimiento o QA.
2. Dataset documentado.
3. Modelo base.
4. Evaluación offline.
5. API de inferencia.
6. Pruebas funcionales.
7. Registro de métricas operativas.
8. Documentación de despliegue.

Métricas sugeridas:

1. Accuracy, precision, recall y F1.
2. Latencia promedio.
3. Tasa de error.
4. Throughput básico.
5. Comparación entre modelo base y variante.
6. Pruebas con entradas nuevas.

Despliegue mínimo:

1. API o interfaz funcional.
2. Entrada nueva.
3. Salida verificable.
4. Instrucciones reproducibles.
5. Evidencia de operación.

No será suficiente ejecutar un notebook. El estudiante debe demostrar que el sistema puede utilizarse como servicio.



### Ajuste de rúbrica por repositorio

La rúbrica principal se mantiene sobre 20 puntos:

1. Cuaderno técnico reproducible: 3 puntos.
2. Despliegue funcional: 2 puntos.
3. Video técnico de más de 10 minutos: 5 puntos.
4. Exposición oral y defensa técnica: 10 puntos.

El repositorio no reemplaza ninguno de estos componentes. El repositorio será usado como evidencia transversal para validar:

1. Reproducibilidad del cuaderno.
2. Resultados presentados.
3. Despliegue.
4. Proceso de trabajo.
5. Autoría técnica.
6. Historial de avances.
7. Coherencia entre código, video y exposición.

Cuando el repositorio no exista, esté incompleto o no permita verificar resultados, se afectarán directamente los puntos de cuaderno, despliegue, video y exposición, según corresponda.



### Criterio estricto de evaluación

El estudiante debe demostrar que entiende su proyecto.

Se penalizará especialmente:

1. Código generado que el estudiante no puede explicar.
2. Fórmulas, símbolos o diagramas sin interpretación.
3. Uso decorativo de términos como RAG, agente, embedding, MLOps, LLMOps, fine-tuning o multimodalidad.
4. Ausencia de línea base.
5. Ausencia de métricas.
6. Ausencia de análisis de errores.
7. Repositorio subido al final sin evolución.
8. Despliegue que no puede probarse.
9. Cuaderno con código sin explicación.
10. Exposición basada en lectura mecánica.

Un proyecto pequeño, específico, evaluado y bien defendido puede obtener mayor nota que un proyecto complejo que el estudiante no entiende.
