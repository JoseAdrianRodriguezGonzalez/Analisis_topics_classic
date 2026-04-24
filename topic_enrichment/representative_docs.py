'''
representative_docs.py
----------------------
Identifica los documentos más representativos de cada cluster
calculando la similitud coseno entre cada documento y el centroide
del cluster en el espacio vectorial dado.

El centroide se define como el promedio aritmético de los vectores
de todos los documentos del cluster. Los documentos con mayor
similitud coseno al centroide son los que mejor "resumen" el cluster.

Lógica general:
    1. Para cada cluster, calcular el centroide promediando sus vectores
    2. Medir similitud coseno de cada documento con el centroide
    3. Ordenar por similitud descendente
    4. Retornar los top K documentos con sus metadatos e índices originales

Funciones públicas:
    compute_centroids          -- calcula el centroide de cada cluster
    cosine_similarity_to_centroid  -- similitud coseno documento-centroide
    get_representative_docs    -- retorna los K docs más representativos por cluster
    representative_docs_to_dataframe -- convierte resultados a DataFrame tabular
'''

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_centroids(
    X: np.ndarray,
    labels: list[int] | np.ndarray,
    exclude_noise: bool = True,
) -> dict[int, np.ndarray]:
    '''
    Calcula el centroide (promedio aritmético) de cada cluster
    en el espacio vectorial de X.

    Parámetros:
        X             -- matriz de vectores (n_docs, n_dims)
        labels        -- etiquetas de cluster por documento (-1 = ruido)
        exclude_noise -- si True, ignora documentos con etiqueta -1

    Retorna:
        Diccionario {cluster_id: vector_centroide}
    '''
    labels_array = np.array(labels)
    cluster_ids = sorted(set(labels_array))

    if exclude_noise:
        cluster_ids = [c for c in cluster_ids if c != -1]

    centroides: dict[int, np.ndarray] = {}

    for cluster_id in cluster_ids:
        mascara = labels_array == cluster_id
        vectores_cluster = X[mascara]

        if len(vectores_cluster) == 0:
            logger.warning('Cluster %d: sin vectores para calcular centroide', cluster_id)
            continue

        # Centroide como promedio de todos los vectores del cluster
        centroide = vectores_cluster.mean(axis=0)
        centroides[cluster_id] = centroide

        logger.debug(
            'Cluster %d: centroide calculado desde %d documentos (dims=%d)',
            cluster_id, len(vectores_cluster), centroide.shape[0]
        )

    logger.info('Centroides calculados para %d clusters', len(centroides))
    return centroides


def _cosine_similarity_vector(
    vector: np.ndarray,
    centroide: np.ndarray,
) -> float:
    '''
    Calcula la similitud coseno entre un vector y el centroide del cluster.
    Retorna 0.0 si alguno de los dos tiene norma cero.

    Similitud coseno = (v · c) / (||v|| * ||c||)
    '''
    norma_vector = np.linalg.norm(vector)
    norma_centroide = np.linalg.norm(centroide)

    if norma_vector == 0.0 or norma_centroide == 0.0:
        return 0.0

    return float(np.dot(vector, centroide) / (norma_vector * norma_centroide))


def cosine_similarity_to_centroid(
    X: np.ndarray,
    centroide: np.ndarray,
    indices_cluster: np.ndarray,
) -> list[tuple[int, float]]:
    '''
    Calcula la similitud coseno de cada documento de un cluster con su centroide.

    Parámetros:
        X                -- matriz de vectores completa (n_docs_total, n_dims)
        centroide        -- vector centroide del cluster (n_dims,)
        indices_cluster  -- índices originales de los documentos del cluster en X

    Retorna:
        Lista de tuplas (indice_original, similitud_coseno) ordenada
        por similitud descendente
    '''
    similitudes = []

    for idx_original in indices_cluster:
        vector = X[idx_original]
        similitud = _cosine_similarity_vector(vector, centroide)
        similitudes.append((int(idx_original), similitud))

    # Ordenar por similitud descendente
    similitudes.sort(key=lambda x: x[1], reverse=True)
    return similitudes


def get_representative_docs(
    X: np.ndarray,
    labels: list[int] | np.ndarray,
    corpus: list[str],
    top_k: int = 5,
    exclude_noise: bool = True,
    metadata: pd.DataFrame | None = None,
) -> dict[int, list[dict]]:
    '''
    Retorna los K documentos más representativos de cada cluster,
    medidos por similitud coseno al centroide del cluster.

    Parámetros:
        X             -- matriz de vectores (n_docs, n_dims)
        labels        -- etiquetas de cluster por documento
        corpus        -- lista de textos originales (alineada con X y labels)
        top_k         -- número de documentos representativos a retornar por cluster
        exclude_noise -- si True, ignora documentos con etiqueta -1
        metadata      -- DataFrame opcional con metadatos por documento
                         (debe estar alineado con corpus y labels por índice)

    Retorna:
        Diccionario {cluster_id: [{'indice_original': int,
                                   'similitud_coseno': float,
                                   'texto': str,
                                   'rank': int,
                                   ...metadatos opcionales}, ...]}
    '''
    labels_array = np.array(labels)
    corpus_array = np.array(corpus, dtype=object)

    # Verificar alineación
    if len(X) != len(labels_array) or len(X) != len(corpus_array):
        raise ValueError(
            f'X ({len(X)}), labels ({len(labels_array)}) y corpus ({len(corpus_array)}) '
            f'deben tener el mismo número de filas'
        )

    # Calcular centroides
    centroides = compute_centroids(X, labels_array, exclude_noise)

    cluster_ids = sorted(centroides.keys())
    resultados: dict[int, list[dict]] = {}

    for cluster_id in cluster_ids:
        centroide = centroides[cluster_id]
        mascara = labels_array == cluster_id
        indices_cluster = np.where(mascara)[0]

        # Calcular similitudes con el centroide
        similitudes_ordenadas = cosine_similarity_to_centroid(
            X, centroide, indices_cluster
        )

        # Tomar los top K más similares
        top_similitudes = similitudes_ordenadas[:top_k]
        docs_representativos = []

        for rank, (idx_original, similitud) in enumerate(top_similitudes, start=1):
            entrada: dict = {
                'indice_original' : idx_original,
                'rank'            : rank,
                'similitud_coseno': round(similitud, 6),
                'texto'           : str(corpus_array[idx_original]),
                'n_docs_cluster'  : int(mascara.sum()),
            }

            # Agregar metadatos si están disponibles
            if metadata is not None and idx_original < len(metadata):
                for columna in metadata.columns:
                    entrada[columna] = metadata.iloc[idx_original][columna]

            docs_representativos.append(entrada)

        resultados[cluster_id] = docs_representativos

        logger.debug(
            'Cluster %d (%d docs): doc más representativo en índice %d (similitud=%.4f)',
            cluster_id,
            int(mascara.sum()),
            top_similitudes[0][0] if top_similitudes else -1,
            top_similitudes[0][1] if top_similitudes else 0.0,
        )

    logger.info(
        'Documentos representativos extraídos para %d clusters (top_k=%d)',
        len(resultados), top_k
    )
    return resultados


def representative_docs_to_dataframe(
    docs_por_cluster: dict[int, list[dict]],
) -> pd.DataFrame:
    '''
    Convierte el diccionario de documentos representativos a un DataFrame
    tabular para facilitar la exportación a CSV.

    Parámetros:
        docs_por_cluster -- output de get_representative_docs

    Retorna:
        DataFrame con columnas: cluster_id, rank, indice_original,
        similitud_coseno, n_docs_cluster, texto, [columnas de metadata]
    '''
    filas = []
    for cluster_id, documentos in docs_por_cluster.items():
        for doc in documentos:
            fila = {'cluster_id': cluster_id}
            fila.update(doc)
            filas.append(fila)

    df = pd.DataFrame(filas)

    # Reordenar columnas para que cluster_id y rank queden primero
    columnas_base = ['cluster_id', 'rank', 'indice_original',
                     'similitud_coseno', 'n_docs_cluster', 'texto']
    columnas_extra = [c for c in df.columns if c not in columnas_base]
    df = df[columnas_base + columnas_extra]

    logger.info(
        'DataFrame de documentos representativos generado: %d filas (%d clusters)',
        len(df), df['cluster_id'].nunique() if not df.empty else 0
    )
    return df