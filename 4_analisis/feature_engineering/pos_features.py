'''
pos_features.py
---------------
Calcula la distribucion de partes del discurso (POS tags) por documento
usando el modelo de spaCy en espanol.

Las distribuciones se expresan como proporciones (valor entre 0 y 1)
del total de tokens con POS asignado en cada comentario, lo que hace
las features comparables entre comentarios de distinta longitud.

POS tags incluidas:
    NOUN  -- sustantivos
    VERB  -- verbos
    ADJ   -- adjetivos
    ADV   -- adverbios
    PROPN -- nombres propios

Funciones:
    load_spacy_model    -- carga el modelo es_core_news_sm con manejo de errores
    tag_document        -- aplica POS tagging a un texto y retorna conteos
    compute_pos_features -- aplica tagging a todo el corpus y retorna DataFrame
'''

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tags que nos interesan para el analisis de turismo
RELEVANT_POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']


def load_spacy_model(model_name: str = 'es_core_news_sm'):
    '''
    Carga el modelo de spaCy indicado. Si no esta instalado, lo descarga
    automaticamente y reintenta la carga.

    Retorna el objeto nlp de spaCy listo para usar.
    '''
    try:
        import spacy
        nlp = spacy.load(model_name)
        logger.info('Modelo spaCy cargado: %s', model_name)
        return nlp

    except OSError:
        logger.warning('Modelo %s no encontrado, descargando...', model_name)
        os.system(f'python -m spacy download {model_name}')

        import spacy
        nlp = spacy.load(model_name)
        logger.info('Modelo spaCy descargado y cargado: %s', model_name)
        return nlp


def tag_document(text: str, nlp) -> dict[str, int]:
    '''
    Aplica POS tagging a un documento y retorna un diccionario con
    el conteo de cada tag relevante.

    Documentos vacios o nulos retornan conteos en cero.

    Parametros:
        text -- comentario limpio como string
        nlp  -- modelo de spaCy ya cargado

    Retorna:
        Diccionario {tag: conteo} para los tags en RELEVANT_POS_TAGS.
    '''
    counts = {tag: 0 for tag in RELEVANT_POS_TAGS}

    if not text or not isinstance(text, str) or not text.strip():
        return counts

    doc = nlp(text)
    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1

    return counts


def compute_pos_features(
    cleaned_series: pd.Series,
    nlp,
) -> pd.DataFrame:
    '''
    Aplica POS tagging a todos los documentos del corpus y calcula la
    proporcion de cada tag relevante respecto al total de tokens etiquetados.

    La proporcion se calcula solo sobre los tokens que tienen alguno de
    los tags relevantes, no sobre el total de tokens del comentario, para
    que las distribuciones sumen aproximadamente 1 y sean directamente
    comparables.

    Columnas de salida (una por tag en RELEVANT_POS_TAGS):
        pos_ratio_noun  -- proporcion de sustantivos
        pos_ratio_verb  -- proporcion de verbos
        pos_ratio_adj   -- proporcion de adjetivos
        pos_ratio_adv   -- proporcion de adverbios
        pos_ratio_propn -- proporcion de nombres propios

    Tambien incluye:
        pos_total_tagged -- total de tokens etiquetados con tags relevantes
    '''
    rows = []

    for text in cleaned_series.fillna(''):
        counts = tag_document(text, nlp)
        rows.append(counts)

    counts_df = pd.DataFrame(rows, index=cleaned_series.index)

    # Total de tokens etiquetados con alguno de los tags relevantes
    total_tagged = counts_df.sum(axis=1)

    # Proporciones: si el documento no tiene ningun tag relevante, ratio = 0
    ratio_df = pd.DataFrame(index=cleaned_series.index)
    for tag in RELEVANT_POS_TAGS:
        col_name = f'pos_ratio_{tag.lower()}'
        ratio_df[col_name] = np.where(
            total_tagged > 0,
            counts_df[tag] / total_tagged,
            0.0,
        ).round(6)

    ratio_df['pos_total_tagged'] = total_tagged.to_numpy(dtype=int)

    logger.info(
        'POS features calculadas: %d documentos, promedio etiquetados por doc=%.2f',
        len(ratio_df),
        ratio_df['pos_total_tagged'].mean(),
    )
    return ratio_df
