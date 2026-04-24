'''
text_features.py
----------------
Calcula features basicas derivadas de la longitud y composicion
textual de cada comentario limpio.

Funciones:
    count_tokens        -- numero de tokens por documento
    count_characters    -- numero de caracteres por documento
    compute_text_length_features -- combina ambas en un DataFrame
'''

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def count_tokens(cleaned_series: pd.Series) -> np.ndarray:
    '''
    Cuenta el numero de tokens (palabras) en cada comentario.
    Los comentarios nulos o vacios devuelven 0.
    '''
    token_counts = (
        cleaned_series
        .fillna('')
        .apply(lambda text: len(text.split()) if text.strip() else 0)
        .to_numpy(dtype=int)
    )
    logger.debug('Conteo de tokens completado para %d documentos', len(token_counts))
    return token_counts


def count_characters(cleaned_series: pd.Series) -> np.ndarray:
    '''
    Cuenta el numero de caracteres (sin espacios) en cada comentario.
    Los comentarios nulos o vacios devuelven 0.
    '''
    char_counts = (
        cleaned_series
        .fillna('')
        .apply(lambda text: len(text.replace(' ', '')) if text.strip() else 0)
        .to_numpy(dtype=int)
    )
    logger.debug('Conteo de caracteres completado para %d documentos', len(char_counts))
    return char_counts


def compute_text_length_features(cleaned_series: pd.Series) -> pd.DataFrame:
    '''
    Combina conteo de tokens, caracteres y longitud promedio de token
    en un DataFrame alineado con el indice de entrada.

    Columnas de salida:
        token_count      -- numero de tokens
        char_count       -- numero de caracteres sin espacios
        avg_token_length -- promedio de longitud de token (char_count / token_count)
    '''
    token_counts = count_tokens(cleaned_series)
    char_counts  = count_characters(cleaned_series)

    # Evita division por cero cuando el comentario esta vacio
    avg_token_length = np.where(
        token_counts > 0,
        char_counts / token_counts,
        0.0
    )

    result = pd.DataFrame(
        {
            'token_count'     : token_counts,
            'char_count'      : char_counts,
            'avg_token_length': np.round(avg_token_length, 4),
        },
        index=cleaned_series.index,
    )

    logger.info(
        'Features de longitud calculadas: %d documentos, token promedio %.2f',
        len(result),
        result['token_count'].mean(),
    )
    return result
