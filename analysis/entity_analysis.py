'''
entity_analysis.py
------------------
Bloque 8 — Análisis basado en entidades nombradas (NER).

Cruza los grupos de entidades producidos por el pipeline de Adrian
(ner_groups.json) con el corpus enriquecido de sentimiento para
responder: ¿qué se dice de cada entidad y con qué valoración?

Fuentes de entrada:
    ner_groups.json            -- entidades agrupadas con índices de documentos
    corpus_con_sentimiento.csv -- corpus con topic, location, sentimiento
    docs_with_topics.csv       -- topic y location por documento

Salidas:
    entidades_con_sentimiento.csv  -- sentimiento promedio por entidad
    entidades_por_destino.csv      -- entidades más mencionadas por location
    entidades_por_topico.csv       -- entidades más presentes por tópico BERTopic

Lógica:
    Para cada entidad, los índices de documentos que la mencionan se usan
    para calcular el sentimiento promedio, el tópico más frecuente y la
    distribución por destino. Esto permite saber, por ejemplo, que "playa"
    aparece en 800 documentos con sentimiento medio 0.6, concentrada
    en riviera_maya, asociada principalmente al tópico 0 (snorkel).
'''

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================
# RUTAS
# ======================================================

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'analysis' / 'entities'

PATH_NER_GROUPS    = BASE_DIR /'data'  / 'features' / 'ner_groups.json'
PATH_DOCS_TOPICS   = BASE_DIR  /'data'/ 'results' / 'docs_with_topics.csv'

# Este archivo es generado por el paso 1 (sentiment_analysis), así que se queda en data/
PATH_SENTIMENT     = DATA_DIR / 'analysis' / 'sentiment' / 'corpus_con_sentimiento.csv'

# Número mínimo de documentos para incluir una entidad en el análisis
MIN_DOCS_ENTIDAD = 5

# Número máximo de entidades a exportar (ordenadas por frecuencia)
MAX_ENTIDADES = 500


# ======================================================
# CARGA
# ======================================================

def _cargar_ner_groups(path: Path) -> list[dict]:
    '''
    Carga el archivo ner_groups.json producido por el pipeline de Adrian.

    Cada elemento tiene: text, label, count, indices,
    top_noun_phrases, noun_phrases_freq.
    '''
    with open(path, encoding='utf-8') as archivo:
        grupos = json.load(archivo)
    logger.info('NER groups cargados: %d entidades', len(grupos))
    return grupos


def _cargar_corpus_sentimiento(path_sentimiento: Path, path_docs: Path) -> pd.DataFrame:
    '''
    Carga el corpus con sentimiento. Si el archivo no existe (aún no se
    ejecutó sentiment_analysis.py), carga el corpus base sin sentimiento.
    '''
    if path_sentimiento.exists():
        df = pd.read_csv(path_sentimiento)
        logger.info('Corpus con sentimiento cargado: %d documentos', len(df))
    else:
        logger.warning(
            'corpus_con_sentimiento.csv no encontrado en %s. '
            'Ejecuta run_sentiment_analysis() primero. '
            'Usando corpus base sin sentimiento.',
            path_sentimiento,
        )
        df = pd.read_csv(path_docs)
        df['sentimiento_numerico'] = np.nan
        df['sentimiento_estrella'] = 'sin_etiqueta'
        df['estrellas'] = np.nan

    return df


# ======================================================
# ENRIQUECIMIENTO DE ENTIDADES
# ======================================================

def _enriquecer_entidad(
    entidad: dict,
    df_corpus: pd.DataFrame,
    indice_a_fila: dict[int, int],
) -> dict:
    '''
    Para una entidad dada, recupera los documentos que la mencionan
    y calcula métricas de sentimiento, distribución por destino y tópico.

    Parámetros:
        entidad        -- dict con text, label, count, indices
        df_corpus      -- DataFrame del corpus con sentimiento
        indice_a_fila  -- mapeo {indice_documento: posición_en_df}

    Retorna:
        dict con métricas enriquecidas o None si la entidad tiene
        menos documentos que MIN_DOCS_ENTIDAD.
    '''
    indices_validos = [
        idx for idx in entidad['indices']
        if idx in indice_a_fila
    ]

    if len(indices_validos) < MIN_DOCS_ENTIDAD:
        return None

    filas = df_corpus.iloc[[indice_a_fila[idx] for idx in indices_validos]]

    # Sentimiento
    sentimiento_values = filas['sentimiento_numerico'].dropna()
    sentimiento_medio  = round(sentimiento_values.mean(), 4) if len(sentimiento_values) > 0 else np.nan
    n_con_rating       = int(sentimiento_values.notna().sum()) if 'sentimiento_numerico' in filas else 0

    # Distribución por categoría de sentimiento
    if 'sentimiento_estrella' in filas.columns:
        dist_sentimiento = filas['sentimiento_estrella'].value_counts().to_dict()
    else:
        dist_sentimiento = {}

    # Estrellas medias (solo documentos con rating)
    if 'estrellas' in filas.columns:
        estrella_media = round(filas['estrellas'].mean(), 3) if filas['estrellas'].notna().any() else np.nan
    else:
        estrella_media = np.nan

    # Distribución por destino
    if 'location' in filas.columns:
        dist_location = filas['location'].value_counts().to_dict()
        destino_principal = filas['location'].value_counts().index[0] if len(filas) > 0 else None
    else:
        dist_location  = {}
        destino_principal = None

    # Distribución por tópico
    if 'topic' in filas.columns:
        topicos_validos = filas[filas['topic'] != -1]['topic']
        if len(topicos_validos) > 0:
            topico_principal = int(topicos_validos.value_counts().index[0])
            pct_ruido = round((filas['topic'] == -1).sum() / len(filas) * 100, 2)
        else:
            topico_principal = -1
            pct_ruido = 100.0
    else:
        topico_principal = None
        pct_ruido = np.nan

    return {
        'entidad'             : entidad['text'],
        'label'               : entidad['label'],
        'n_documentos'        : len(indices_validos),
        'n_con_rating'        : n_con_rating,
        'sentimiento_medio'   : sentimiento_medio,
        'estrella_media'      : estrella_media,
        'n_negativo'          : dist_sentimiento.get('negativo', 0),
        'n_neutro'            : dist_sentimiento.get('neutro', 0),
        'n_positivo'          : dist_sentimiento.get('positivo', 0),
        'n_sin_etiqueta'      : dist_sentimiento.get('sin_etiqueta', 0),
        'destino_principal'   : destino_principal,
        'topico_principal'    : topico_principal,
        'pct_ruido_topico'    : pct_ruido,
        'dist_location'       : str(dist_location),
    }


def calcular_entidades_con_sentimiento(
    ner_grupos: list[dict],
    df_corpus: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Calcula métricas de sentimiento para todas las entidades del corpus.

    Retorna DataFrame con una fila por entidad, ordenado por n_documentos
    descendente.
    '''
    # Construir índice de posición en el DataFrame para búsqueda rápida
    indice_a_fila = {
        int(row.indice): pos
        for pos, row in enumerate(df_corpus.itertuples(index=False))
        if hasattr(row, 'indice')
    }

    logger.info(
        'Procesando %d entidades (mínimo %d documentos)...',
        len(ner_grupos),
        MIN_DOCS_ENTIDAD,
    )

    resultados = []
    for i, entidad in enumerate(ner_grupos):
        if i > 0 and i % 500 == 0:
            logger.debug('  Procesadas %d / %d entidades', i, len(ner_grupos))

        resultado = _enriquecer_entidad(entidad, df_corpus, indice_a_fila)
        if resultado is not None:
            resultados.append(resultado)

    df_entidades = pd.DataFrame(resultados)
    df_entidades = df_entidades.sort_values('n_documentos', ascending=False).reset_index(drop=True)

    logger.info(
        'Entidades procesadas: %d de %d superaron el umbral de %d documentos',
        len(df_entidades),
        len(ner_grupos),
        MIN_DOCS_ENTIDAD,
    )
    return df_entidades


# ======================================================
# AGRUPACIONES SECUNDARIAS
# ======================================================

def _entidades_por_destino(
    ner_grupos: list[dict],
    df_corpus: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    '''
    Para cada destino, calcula las top_n entidades más mencionadas
    con su sentimiento promedio.

    Retorna DataFrame con columnas: location, entidad, label,
    n_documentos, sentimiento_medio.
    '''
    indice_a_location = dict(zip(df_corpus['indice'], df_corpus['location']))
    indice_a_sentimiento = (
        dict(zip(df_corpus['indice'], df_corpus['sentimiento_numerico']))
        if 'sentimiento_numerico' in df_corpus.columns
        else {}
    )

    filas = []
    for entidad in ner_grupos:
        conteo_por_destino: Counter = Counter()
        sentimiento_por_destino: dict[str, list] = {}

        for idx in entidad['indices']:
            loc = indice_a_location.get(idx)
            if loc is None:
                continue
            conteo_por_destino[loc] += 1
            sent = indice_a_sentimiento.get(idx, np.nan)
            if not np.isnan(sent) if isinstance(sent, float) else True:
                sentimiento_por_destino.setdefault(loc, []).append(sent)

        for loc, n in conteo_por_destino.items():
            if n < MIN_DOCS_ENTIDAD:
                continue
            valores_sent = [v for v in sentimiento_por_destino.get(loc, []) if not np.isnan(v)]
            filas.append({
                'location'          : loc,
                'entidad'           : entidad['text'],
                'label'             : entidad['label'],
                'n_documentos'      : n,
                'sentimiento_medio' : round(np.mean(valores_sent), 4) if valores_sent else np.nan,
            })

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    df = (
        df.sort_values(['location', 'n_documentos'], ascending=[True, False])
        .groupby('location')
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info('Entidades por destino calculadas: %d filas', len(df))
    return df


def _entidades_por_topico(
    ner_grupos: list[dict],
    df_corpus: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    '''
    Para cada tópico BERTopic (excluyendo ruido -1), calcula las top_n
    entidades más mencionadas con su sentimiento promedio.
    '''
    if 'topic' not in df_corpus.columns:
        logger.warning('Columna topic no disponible. Saltando entidades por tópico.')
        return pd.DataFrame()

    indice_a_topico = dict(zip(df_corpus['indice'], df_corpus['topic']))
    indice_a_sentimiento = (
        dict(zip(df_corpus['indice'], df_corpus['sentimiento_numerico']))
        if 'sentimiento_numerico' in df_corpus.columns
        else {}
    )

    filas = []
    for entidad in ner_grupos:
        conteo_por_topico: Counter = Counter()
        sentimiento_por_topico: dict[int, list] = {}

        for idx in entidad['indices']:
            topico = indice_a_topico.get(idx, -1)
            if topico == -1:
                continue
            conteo_por_topico[topico] += 1
            sent = indice_a_sentimiento.get(idx, np.nan)
            if isinstance(sent, float) and not np.isnan(sent):
                sentimiento_por_topico.setdefault(topico, []).append(sent)

        for topico, n in conteo_por_topico.items():
            if n < MIN_DOCS_ENTIDAD:
                continue
            valores_sent = sentimiento_por_topico.get(topico, [])
            filas.append({
                'topic'             : topico,
                'entidad'           : entidad['text'],
                'label'             : entidad['label'],
                'n_documentos'      : n,
                'sentimiento_medio' : round(np.mean(valores_sent), 4) if valores_sent else np.nan,
            })

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    df = (
        df.sort_values(['topic', 'n_documentos'], ascending=[True, False])
        .groupby('topic')
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info('Entidades por tópico calculadas: %d filas', len(df))
    return df


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def run_entity_analysis(
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    '''
    Pipeline completo de análisis basado en entidades NER.

    Retorna diccionario con los tres DataFrames producidos:
        entidades, por_destino, por_topico
    '''
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('=== Iniciando análisis de entidades ===')

    if not PATH_NER_GROUPS.exists():
        logger.error('ner_groups.json no encontrado en %s — pipeline abortado', PATH_NER_GROUPS)
        return {}

    ner_grupos  = _cargar_ner_groups(PATH_NER_GROUPS)
    df_corpus   = _cargar_corpus_sentimiento(PATH_SENTIMENT, PATH_DOCS_TOPICS)

    # Análisis principal: entidades con sentimiento
    df_entidades = calcular_entidades_con_sentimiento(
        ner_grupos[:MAX_ENTIDADES],  # limitar a las más frecuentes
        df_corpus,
    )
    df_entidades.to_csv(
        output_dir / 'entidades_con_sentimiento.csv',
        index=False, encoding='utf-8-sig',
    )

    # Agrupación por destino
    df_por_destino = _entidades_por_destino(ner_grupos[:MAX_ENTIDADES], df_corpus)
    df_por_destino.to_csv(
        output_dir / 'entidades_por_destino.csv',
        index=False, encoding='utf-8-sig',
    )

    # Agrupación por tópico
    df_por_topico = _entidades_por_topico(ner_grupos[:MAX_ENTIDADES], df_corpus)
    df_por_topico.to_csv(
        output_dir / 'entidades_por_topico.csv',
        index=False, encoding='utf-8-sig',
    )

    logger.info('=== Análisis de entidades completado. Archivos en: %s ===', output_dir)

    return {
        'entidades'    : df_entidades,
        'por_destino'  : df_por_destino,
        'por_topico'   : df_por_topico,
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_entity_analysis()
