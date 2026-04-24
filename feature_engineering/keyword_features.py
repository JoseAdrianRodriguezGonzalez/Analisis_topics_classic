'''
keyword_features.py
-------------------
Calcula features relacionadas con la presencia y frecuencia de
palabras clave (vocabulario TF-IDF) dentro de cada comentario.

Las funciones de vectorizacion (generar_ngrams, calcular_BoW) son
copias directas de 3_analisis/model/Vectorization.py para evitar
importar ese modulo, cuyo codigo de ejecucion a nivel de modulo
requiere rutas relativas incompatibles con este proyecto.

Funciones:
    load_vocabulary          -- carga el vocabulario desde un CSV de rankings
    compute_keyword_presence -- cuenta keywords presentes por documento
    compute_keyword_density  -- keywords presentes / total tokens
    compute_keyword_features -- combina ambas en un DataFrame
'''

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Funciones de vectorizacion
# Copiadas de 3_analisis/model/Vectorization.py
# Se mantienen aqui para evitar dependencia de ese modulo
# ------------------------------------------------------

def _generar_ngrams(palabras: list[str], n: int) -> list[str]:
    '''
    Dado una lista de palabras y n, devuelve lista de n-gramas como strings.
    Ejemplo: ["a", "b", "c"], n=2 -> ["a b", "b c"]
    '''
    return [' '.join(palabras[i:i + n]) for i in range(len(palabras) - n + 1)]


def _calcular_bow(corpus: list[str], vocabulario: list[str], n: int = 1):
    '''
    Construye la matriz Bag of Words para el corpus dado el vocabulario.
    Retorna la matriz BoW filtrada (sin filas de ceros), los indices vacios
    y la mascara booleana de filas validas.
    '''
    vocab_index = {palabra: idx for idx, palabra in enumerate(vocabulario)}
    bow_matrix  = np.zeros((len(corpus), len(vocabulario)), dtype=int)

    empty_indices = []
    for i, documento in enumerate(corpus):
        if pd.isna(documento):
            empty_indices.append(i)
            continue
        palabras = documento.split()
        tokens   = _generar_ngrams(palabras, n)
        for token in tokens:
            if token in vocab_index:
                bow_matrix[i, vocab_index[token]] += 1

    valid_mask = bow_matrix.sum(axis=1) > 0
    return bow_matrix[valid_mask], empty_indices, valid_mask


# ------------------------------------------------------
# Funciones propias del modulo
# ------------------------------------------------------

def load_vocabulary(ranking_csv_path: str | Path) -> list[str]:
    '''
    Carga la lista de ngrams desde un archivo de rankings generado
    por nlp_analysis.py (columna "ngram").

    Parametros:
        ranking_csv_path -- ruta al CSV (rankings_unigrams.csv, etc.)
    '''
    df = pd.read_csv(ranking_csv_path)

    if 'ngram' not in df.columns:
        raise ValueError(
            f'El archivo {ranking_csv_path} no contiene la columna "ngram".'
        )

    vocabulary = df['ngram'].dropna().tolist()
    logger.info('Vocabulario cargado: %d terminos desde %s', len(vocabulary), ranking_csv_path)
    return vocabulary


def compute_keyword_presence(
    corpus: list[str],
    vocabulary: list[str],
    ngram_n: int,
) -> np.ndarray:
    '''
    Cuenta cuantas palabras del vocabulario aparecen (al menos una vez)
    en cada documento.

    El resultado esta alineado con el corpus original: los documentos
    vacios o nulos reciben valor 0.

    Parametros:
        corpus     -- lista de comentarios limpios
        vocabulary -- lista de ngrams del vocabulario
        ngram_n    -- 1 para unigramas, 2 para bigramas, 3 para trigramas
    '''
    if not corpus:
        logger.warning('Corpus vacio recibido en compute_keyword_presence')
        return np.array([], dtype=int)

    bow_matrix, _, valid_mask = _calcular_bow(corpus, vocabulary, n=ngram_n)

    # Reconstruye el vector completo alineado con el corpus original
    # Los documentos fuera de valid_mask quedan en 0
    keyword_presence  = np.zeros(len(corpus), dtype=int)
    valid_positions   = np.where(valid_mask)[0]

    for pos_in_bow, original_pos in enumerate(valid_positions):
        keyword_presence[original_pos] = int((bow_matrix[pos_in_bow] > 0).sum())

    logger.debug(
        'Presencia de keywords calculada: media=%.2f, max=%d',
        keyword_presence.mean(),
        keyword_presence.max(),
    )
    return keyword_presence


def compute_keyword_density(
    keyword_presence: np.ndarray,
    token_counts: np.ndarray,
) -> np.ndarray:
    '''
    Calcula la densidad de keywords como la proporcion de tokens del
    vocabulario respecto al total de tokens del documento.

        densidad = keywords_presentes / total_tokens

    Documentos con cero tokens reciben densidad 0.

    Parametros:
        keyword_presence -- array con conteo de keywords presentes
        token_counts     -- array con total de tokens por documento
    '''
    density = np.where(
        token_counts > 0,
        keyword_presence / token_counts,
        0.0,
    )
    return np.round(density, 6)


def compute_keyword_features(
    corpus: list[str],
    token_counts: np.ndarray,
    vocabulary: list[str],
    ngram_n: int,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    '''
    Combina presencia y densidad de keywords en un DataFrame listo
    para concatenar con el resto de features.

    Columnas de salida:
        keyword_count   -- numero de keywords del vocabulario presentes
        keyword_density -- keyword_count / token_count
    '''
    keyword_presence = compute_keyword_presence(corpus, vocabulary, ngram_n)
    keyword_density  = compute_keyword_density(keyword_presence, token_counts)

    result = pd.DataFrame(
        {
            'keyword_count'  : keyword_presence,
            'keyword_density': keyword_density,
        },
        index=index,
    )

    logger.info(
        'Features de keywords calculadas: densidad media=%.4f',
        result['keyword_density'].mean(),
    )
    return result