Plan de trabajo – Análisis de respuestas abiertas de la encuesta
1. Objetivo
Analizar las respuestas abiertas de la encuesta para identificar patrones temáticos, agrupaciones semánticas y diferencias según variables demográficas (edad, procedencia y género).
El análisis combinará procesamiento de lenguaje natural (NLP), clustering y modelos de tópicos (LDA).
2. Preparación de los datos
2.1 Variables a considerar
Se trabajará con las siguientes variables de la encuesta:
Edad → agrupar en rangos (ej. 18–25, 26–35, etc.)
Procedencia
Género
Pregunta abierta (texto a analizar)}
Esto permitirá comparar patrones de lenguaje entre grupos.
2.2 Limpieza del texto
Procesamiento inicial de las respuestas abiertas:
Normalización del texto
pasar a minúsculas
eliminar signos de puntuación
Eliminación de stopwords
Tokenización
(Opcional) lematización o stemming
Verificar con nube de palabras la limpieza
3. Representación del texto
3.1 Bolsa de palabras (Bag of Words)
Construir representaciones del texto mediante:
Unigramas
Bigramas
Trigramas
Calcular:
frecuencia total
frecuencia relativa
Esto permitirá identificar expresiones frecuentes y patrones lingüísticos.
4. Análisis exploratorio
4.1 Frecuencias
Generar:
ranking de palabras más frecuentes
ranking de bigramas
ranking de trigramas
Esto ayudará a detectar temas recurrentes en las respuestas.

## Hasta aqui es el preprocesamiento



5. Clustering de documentos
A partir de la matriz de frecuencias:
vectorizar textos (BoW o TF-IDF)
aplicar algoritmo de clustering (ejemplo)
K-means
clustering jerárquico
Objetivo:
identificar grupos de respuestas similares
5.1 Determinar número óptimo de clusters
Usar criterios como:
Silhouette score
Elbow method
Esto permitirá estimar el número adecuado de grupos.
6. Modelado de tópicos
Aplicar Latent Dirichlet Allocation (LDA) para descubrir:
los temas latentes en las respuestas
las palabras más representativas de cada tema
Se evaluará el número de tópicos con:
coherence score
perplexity
7. Análisis interno de cada clúster
Para cada clúster identificado:
calcular TF-IDF
generar un ranking de palabras relevantes
identificar las palabras que caracterizan ese grupo
Esto permitirá interpretar el significado de cada clúster.

8. Identificación de respuestas representativas
Para cada clúster:
calcular el centroide del clúster
medir distancia coseno
identificar los documentos más cercanos al centroide
Estos textos servirán como ejemplos representativos del grupo.
9. Análisis cualitativo asistido con LLM
Dentro de cada clúster:
seleccionar respuestas representativas
enviarlas a un modelo de lenguaje
generar:
síntesis del tema
interpretación semántica
etiquetas conceptuales
Esto permitirá enriquecer el análisis cuantitativo.
10. Visualización
Se generarán visualizaciones como:
nube de palabras
gráfico de frecuencia de n-gramas
visualización de clusters (PCA / t-SNE)
distribución de temas por grupo demográfico
11. Producto final

Se elaborará una infografía que incluya:
principales temas detectados
palabras clave
ejemplos de respuestas