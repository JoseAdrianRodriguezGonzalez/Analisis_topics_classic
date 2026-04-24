'''
keyword_extractor.py
--------------------
Calcula las palabras clave más representativas de cada cluster
usando TF-IDF interno: el vocabulario y las frecuencias se calculan
únicamente sobre los documentos que pertenecen a cada grupo,
no sobre el corpus completo.

Esto garantiza que las palabras extraídas sean características del
cluster y no simplemente las más frecuentes en todo el corpus.

Lógica general:
    1. Agrupar documentos por etiqueta de cluster
    2. Por cada cluster, construir una sub-matriz BoW
    3. Calcular TF por documento dentro del cluster
    4. Calcular IDF usando solo los documentos del cluster
    5. Promediar los scores TF-IDF y ordenar descendentemente
    6. Retornar las top N palabras con su score

Funciones públicas:
    build_vocabulary_from_corpus   -- genera vocabulario desde el corpus limpio
    compute_tfidf_per_cluster      -- calcula TF-IDF interno por cluster
    extract_top_keywords           -- retorna top N keywords por cluster como dict
'''

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_vocabulary_from_corpus(
    corpus: list[str],
    min_freq: int = 2,
    max_vocab_size: int | None = None,
) -> list[str]:
    '''
    Construye el vocabulario de unigramas desde el corpus limpio.
    Filtra términos que aparecen menos de min_freq veces en todo el corpus.

    Parámetros:
        corpus         -- lista de textos limpios (tokenizados por espacio)
        min_freq       -- frecuencia mínima para incluir un término
        max_vocab_size -- límite máximo del vocabulario (None = sin límite)

    Retorna:
        Lista ordenada de términos que conforman el vocabulario
    '''
    # Contar frecuencia global de cada término en el corpus
    counter: Counter = Counter()
    for documento in corpus:
        if not documento or not isinstance(documento, str):
            continue
        tokens = documento.strip().split()
        counter.update(tokens)

    # Filtrar por frecuencia mínima
    vocabulario = [
        termino for termino, frecuencia in counter.items()
        if frecuencia >= min_freq
    ]

    # Ordenar por frecuencia descendente para consistencia
    vocabulario.sort(key=lambda t: counter[t], reverse=True)

    # Aplicar límite si se especificó
    if max_vocab_size is not None:
        vocabulario = vocabulario[:max_vocab_size]

    logger.info(
        'Vocabulario construido: %d terminos (min_freq=%d)',
        len(vocabulario), min_freq
    )
    return vocabulario


def _build_bow_matrix(
    corpus_cluster: list[str],
    vocab_index: dict[str, int],
    vocab_size: int,
) -> np.ndarray:
    '''
    Construye la matriz Bag of Words para un subconjunto del corpus.

    Parámetros:
        corpus_cluster -- lista de textos del cluster
        vocab_index    -- diccionario {termino: índice_columna}
        vocab_size     -- tamaño total del vocabulario

    Retorna:
        Matriz numpy de shape (n_docs_cluster, vocab_size) con conteos enteros
    '''
    n_docs = len(corpus_cluster)
    bow = np.zeros((n_docs, vocab_size), dtype=np.int32)

    for i, documento in enumerate(corpus_cluster):
        if not documento or not isinstance(documento, str):
            continue
        tokens = documento.strip().split()
        for token in tokens:
            if token in vocab_index:
                bow[i, vocab_index[token]] += 1

    return bow


def _calcular_tf(bow: np.ndarray) -> np.ndarray:
    '''
    Calcula la frecuencia de término normalizada por longitud del documento.
    TF(t, d) = conteo(t, d) / total_tokens(d)
    Documentos vacíos reciben TF = 0.
    '''
    totales = bow.sum(axis=1, keepdims=True)
    tf = np.where(totales > 0, bow / totales, 0.0)
    return tf


def _calcular_idf(bow: np.ndarray) -> np.ndarray:
    '''
    Calcula la frecuencia inversa de documento con suavizado.
    IDF(t) = log((N + 1) / (df(t) + 1)) + 1
    donde N es el número de documentos y df(t) es la frecuencia de documento.
    '''
    n_docs = bow.shape[0]
    df = np.sum(bow > 0, axis=0)
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    return idf


def compute_tfidf_per_cluster(
    corpus: list[str],
    labels: list[int] | np.ndarray,
    vocabulary: list[str],
    top_n: int = 15,
    exclude_noise: bool = True,
) -> dict[int, list[dict]]:
    '''
    Calcula TF-IDF interno para cada cluster y retorna los top N términos
    con su score promedio en el cluster.

    Parámetros:
        corpus        -- lista completa de textos limpios (alineada con labels)
        labels        -- etiquetas de cluster por documento (-1 = ruido en HDBSCAN)
        vocabulary    -- vocabulario a usar (output de build_vocabulary_from_corpus)
        top_n         -- número de términos top a retornar por cluster
        exclude_noise -- si True, ignora documentos con etiqueta -1

    Retorna:
        Diccionario {cluster_id: [{'termino': str, 'score': float, 'df': int}, ...]}
        ordenado por score descendente, con top_n entradas por cluster
    '''
    labels_array = np.array(labels)
    corpus_array = np.array(corpus, dtype=object)

    vocab_index = {termino: idx for idx, termino in enumerate(vocabulary)}
    vocab_size = len(vocabulary)

    cluster_ids = sorted(set(labels_array))
    if exclude_noise:
        cluster_ids = [c for c in cluster_ids if c != -1]

    resultados: dict[int, list[dict]] = {}

    for cluster_id in cluster_ids:
        mascara = labels_array == cluster_id
        corpus_cluster = corpus_array[mascara].tolist()
        n_docs_cluster = len(corpus_cluster)

        if n_docs_cluster == 0:
            logger.warning('Cluster %d: sin documentos, omitido', cluster_id)
            continue

        # Construir BoW interno del cluster
        bow = _build_bow_matrix(corpus_cluster, vocab_index, vocab_size)

        # Calcular TF-IDF interno
        tf = _calcular_tf(bow)
        idf = _calcular_idf(bow)
        tfidf = tf * idf

        # Score promedio por término en el cluster
        score_promedio = tfidf.mean(axis=0)

        # Document frequency dentro del cluster (para información adicional)
        df_cluster = (bow > 0).sum(axis=0)

        # Ordenar por score descendente y tomar top N
        indices_top = np.argsort(score_promedio)[::-1][:top_n]

        terminos_top = [
            {
                'termino'      : vocabulary[idx],
                'score_tfidf'  : round(float(score_promedio[idx]), 6),
                'df_en_cluster': int(df_cluster[idx]),
                'n_docs_cluster': n_docs_cluster,
            }
            for idx in indices_top
            if score_promedio[idx] > 0
        ]

        resultados[cluster_id] = terminos_top

        logger.debug(
            'Cluster %d (%d docs): top termino = "%s" (score=%.4f)',
            cluster_id,
            n_docs_cluster,
            terminos_top[0]['termino'] if terminos_top else 'N/A',
            terminos_top[0]['score_tfidf'] if terminos_top else 0.0,
        )

    logger.info(
        'TF-IDF interno calculado para %d clusters (top_n=%d)',
        len(resultados), top_n
    )
    return resultados


def extract_top_keywords(
    corpus: list[str],
    labels: list[int] | np.ndarray,
    top_n: int = 15,
    min_freq: int = 2,
    max_vocab_size: int | None = None,
) -> dict[int, list[dict]]:
    '''
    Función de conveniencia que construye el vocabulario y calcula
    las top keywords por cluster en un solo paso.

    Parámetros:
        corpus         -- lista de textos limpios
        labels         -- etiquetas de cluster por documento
        top_n          -- número de keywords a retornar por cluster
        min_freq       -- frecuencia mínima para incluir un término en el vocabulario
        max_vocab_size -- límite máximo del vocabulario

    Retorna:
        Diccionario {cluster_id: [{'termino': str, 'score_tfidf': float, ...}, ...]}
    '''
    vocabulario = build_vocabulary_from_corpus(corpus, min_freq, max_vocab_size)

    if not vocabulario:
        logger.error('Vocabulario vacío: no se pueden extraer keywords')
        return {}

    return compute_tfidf_per_cluster(corpus, labels, vocabulario, top_n)


def keywords_to_dataframe(
    keywords_por_cluster: dict[int, list[dict]],
) -> pd.DataFrame:
    '''
    Convierte el diccionario de keywords a un DataFrame tabular
    para facilitar la exportación a CSV.

    Parámetros:
        keywords_por_cluster -- output de extract_top_keywords o compute_tfidf_per_cluster

    Retorna:
        DataFrame con columnas: cluster_id, rank, termino, score_tfidf, df_en_cluster, n_docs_cluster
    '''
    filas = []
    for cluster_id, terminos in keywords_por_cluster.items():
        for rank, termino_info in enumerate(terminos, start=1):
            filas.append({
                'cluster_id'    : cluster_id,
                'rank'          : rank,
                'termino'       : termino_info['termino'],
                'score_tfidf'   : termino_info['score_tfidf'],
                'df_en_cluster' : termino_info['df_en_cluster'],
                'n_docs_cluster': termino_info['n_docs_cluster'],
            })

    df = pd.DataFrame(filas)
    logger.info(
        'DataFrame de keywords generado: %d filas (%d clusters)',
        len(df), df['cluster_id'].nunique() if not df.empty else 0
    )
    return df