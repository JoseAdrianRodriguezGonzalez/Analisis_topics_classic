'''
sentiment_analysis.py
---------------------
Bloque 8 — Análisis de sentimiento y emoción a partir de:
    1. Estrellas (rating) como proxy de sentimiento supervisado
    2. Distribución léxica POS como proxy de intensidad emocional
    3. Polaridad por tópico (BERTopic) y por destino

Lógica de sentimiento basada en estrellas:
    1-2  -> negativo
    3    -> neutro
    4-5  -> positivo
    null -> sin_etiqueta (para documentos sin rating, ej. Instagram)

Columnas de salida principales:
    sentimiento_estrella    -- negativo / neutro / positivo / sin_etiqueta
    sentimiento_numerico    -- -1 / 0 / 1 / NaN
    intensidad_adjetivo     -- ratio de adjetivos en el documento (POS)
    intensidad_adverbio     -- ratio de adverbios en el documento (POS)

Agrupaciones exportadas:
    sentimiento_por_topico.csv       -- distribución por tópico BERTopic
    sentimiento_por_destino.csv      -- distribución por location
    sentimiento_por_topico_destino.csv -- cruce tópico x destino

Uso:
    from analysis.sentiment_analysis import run_sentiment_analysis
    run_sentiment_analysis()
'''

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================
# RUTAS
# ======================================================

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'analysis' / 'sentiment'

PATH_DOCS_TOPICS   = BASE_DIR /'data'/ 'results'  / 'docs_with_topics.csv'
PATH_UNIFIED       = BASE_DIR  /'data'/ 'unified'  / 'analysis_unified.csv'
PATH_FEATURES      = BASE_DIR  /'data'/ 'features'  / 'features_nlp.csv'
PATH_TOPICS_META   = BASE_DIR  /'data'/ 'results'  / 'topics.csv'


# ======================================================
# CATEGORIZACIÓN DE SENTIMIENTO
# ======================================================

def _mapear_sentimiento_estrella(estrellas: pd.Series) -> pd.Series:
    '''
    Convierte la columna de estrellas a etiqueta de sentimiento categórico.

    Escala usada:
        1-2  -> negativo
        3    -> neutro
        4-5  -> positivo
        NaN  -> sin_etiqueta
    '''
    def _categorizar(valor):
        if pd.isna(valor):
            return 'sin_etiqueta'
        if valor <= 2:
            return 'negativo'
        if valor == 3:
            return 'neutro'
        return 'positivo'

    return estrellas.apply(_categorizar)


def _mapear_sentimiento_numerico(categoria: pd.Series) -> pd.Series:
    '''
    Convierte la categoría textual a valor numérico [-1, 0, 1, NaN].
    Útil para promedios ponderados.
    '''
    mapa = {
        'negativo'    : -1.0,
        'neutro'      :  0.0,
        'positivo'    :  1.0,
        'sin_etiqueta':  np.nan,
    }
    return categoria.map(mapa)


# ======================================================
# CARGA Y ENSAMBLADO
# ======================================================

def _cargar_corpus_base() -> pd.DataFrame:
    '''
    Carga y ensambla el corpus base unificando:
        - docs_with_topics.csv (topic, location, lang)
        - analysis_unified.csv (estrellas)
        - features_nlp.csv     (pos_ratio_adj, pos_ratio_adv)

    El join se hace por indice. Los campos faltantes se rellenan con NaN.
    '''
    logger.info('Cargando docs_with_topics.csv...')
    df_docs = pd.read_csv(PATH_DOCS_TOPICS)

    logger.info('Cargando analysis_unified.csv...')
    df_unified = pd.read_csv(PATH_UNIFIED, usecols=['indice', 'estrellas'])

    logger.info('Cargando features_nlp.csv...')
    df_features = pd.read_csv(
        PATH_FEATURES,
        usecols=['indice', 'pos_ratio_adj', 'pos_ratio_adv'],
    )

    # Join por indice
    df = df_docs.merge(df_unified, on='indice', how='left')
    df = df.merge(df_features, on='indice', how='left')

    logger.info(
        'Corpus ensamblado: %d documentos | estrellas disponibles: %d',
        len(df),
        df['estrellas'].notna().sum(),
    )
    return df


def _construir_sentimiento(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Añade columnas de sentimiento al DataFrame del corpus.
    '''
    df = df.copy()
    df['sentimiento_estrella']  = _mapear_sentimiento_estrella(df['estrellas'])
    df['sentimiento_numerico']  = _mapear_sentimiento_numerico(df['sentimiento_estrella'])
    df['intensidad_adjetivo']   = df['pos_ratio_adj'].fillna(0.0)
    df['intensidad_adverbio']   = df['pos_ratio_adv'].fillna(0.0)
    return df


# ======================================================
# AGRUPACIONES
# ======================================================

def _sentimiento_por_topico(df: pd.DataFrame, topics_meta: pd.DataFrame) -> pd.DataFrame:
    '''
    Distribución de sentimiento por tópico BERTopic.

    Excluye documentos sin rating (sin_etiqueta) del cálculo de porcentajes
    para no distorsionar la distribución supervisada, pero reporta
    n_sin_etiqueta como columna informativa adicional.
    '''
    # Solo con rating
    df_con_rating = df[df['sentimiento_estrella'] != 'sin_etiqueta'].copy()

    tabla = (
        df_con_rating
        .groupby(['topic', 'sentimiento_estrella'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Asegurarse de que existan las tres columnas de sentimiento
    for col in ['negativo', 'neutro', 'positivo']:
        if col not in tabla.columns:
            tabla[col] = 0

    tabla['total_con_rating'] = tabla[['negativo', 'neutro', 'positivo']].sum(axis=1)

    # Porcentajes
    for col in ['negativo', 'neutro', 'positivo']:
        tabla[f'pct_{col}'] = (
            tabla[col] / tabla['total_con_rating'].replace(0, np.nan) * 100
        ).round(2)

    # Promedio numérico por tópico
    media_num = (
        df_con_rating
        .groupby('topic')['sentimiento_numerico']
        .mean()
        .round(4)
        .rename('sentimiento_medio')
        .reset_index()
    )

    # Documentos sin etiqueta por tópico
    n_sin = (
        df[df['sentimiento_estrella'] == 'sin_etiqueta']
        .groupby('topic')
        .size()
        .rename('n_sin_etiqueta')
        .reset_index()
    )

    tabla = tabla.merge(media_num, on='topic', how='left')
    tabla = tabla.merge(n_sin, on='topic', how='left')
    tabla['n_sin_etiqueta'] = tabla['n_sin_etiqueta'].fillna(0).astype(int)

    # Añadir nombre del tópico desde topics.csv
    if topics_meta is not None:
        tabla = tabla.merge(
            topics_meta[['Topic', 'Name']].rename(columns={'Topic': 'topic', 'Name': 'topic_name'}),
            on='topic',
            how='left',
        )

    tabla = tabla.sort_values('sentimiento_medio', ascending=True).reset_index(drop=True)
    logger.info('Sentimiento por tópico calculado: %d tópicos', len(tabla))
    return tabla


def _sentimiento_por_destino(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Distribución de sentimiento por destino (location).
    Incluye todos los documentos con rating y reporta la media.
    '''
    df_con_rating = df[df['sentimiento_estrella'] != 'sin_etiqueta'].copy()

    tabla = (
        df_con_rating
        .groupby(['location', 'sentimiento_estrella'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ['negativo', 'neutro', 'positivo']:
        if col not in tabla.columns:
            tabla[col] = 0

    tabla['total_con_rating'] = tabla[['negativo', 'neutro', 'positivo']].sum(axis=1)
    tabla['total_documentos']  = df.groupby('location').size().values

    for col in ['negativo', 'neutro', 'positivo']:
        tabla[f'pct_{col}'] = (
            tabla[col] / tabla['total_con_rating'].replace(0, np.nan) * 100
        ).round(2)

    media_num = (
        df_con_rating
        .groupby('location')['sentimiento_numerico']
        .mean()
        .round(4)
        .rename('sentimiento_medio')
        .reset_index()
    )

    estrella_media = (
        df_con_rating
        .groupby('location')['estrellas']
        .mean()
        .round(3)
        .rename('estrella_media')
        .reset_index()
    )

    tabla = tabla.merge(media_num, on='location', how='left')
    tabla = tabla.merge(estrella_media, on='location', how='left')
    tabla = tabla.sort_values('sentimiento_medio', ascending=True).reset_index(drop=True)

    logger.info('Sentimiento por destino calculado: %d destinos', len(tabla))
    return tabla


def _sentimiento_por_topico_destino(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cruce tópico x destino: promedio de sentimiento numérico y
    estrellas medias. Solo documentos con rating.
    Solo se incluyen tópicos válidos (topic != -1).
    '''
    df_valido = df[
        (df['sentimiento_estrella'] != 'sin_etiqueta') &
        (df['topic'] != -1)
    ].copy()

    tabla = (
        df_valido
        .groupby(['topic', 'location'])
        .agg(
            n_docs=('indice', 'count'),
            sentimiento_medio=('sentimiento_numerico', 'mean'),
            estrella_media=('estrellas', 'mean'),
        )
        .round(4)
        .reset_index()
    )

    tabla = tabla.sort_values(['topic', 'sentimiento_medio']).reset_index(drop=True)
    logger.info(
        'Cruce tópico x destino: %d combinaciones',
        len(tabla),
    )
    return tabla


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def run_sentiment_analysis(
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    '''
    Pipeline completo de análisis de sentimiento.

    Retorna diccionario con los tres DataFrames producidos:
        por_topico, por_destino, por_topico_destino
    '''
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('=== Iniciando análisis de sentimiento ===')

    # Cargar metadatos de tópicos (opcional, para añadir nombres)
    topics_meta = None
    if PATH_TOPICS_META.exists():
        topics_meta = pd.read_csv(PATH_TOPICS_META)
        logger.info('Metadatos de tópicos cargados: %d tópicos', len(topics_meta))

    # Ensamblar corpus y añadir columnas de sentimiento
    df = _cargar_corpus_base()
    df = _construir_sentimiento(df)

    logger.info(
        'Distribución de sentimiento global: %s',
        df['sentimiento_estrella'].value_counts().to_dict(),
    )

    # Guardar corpus enriquecido con sentimiento
    path_corpus = output_dir / 'corpus_con_sentimiento.csv'
    df.to_csv(path_corpus, index=False, encoding='utf-8-sig')
    logger.info('Corpus con sentimiento exportado: %s', path_corpus)

    # Calcular agrupaciones
    df_por_topico = _sentimiento_por_topico(df, topics_meta)
    df_por_destino = _sentimiento_por_destino(df)
    df_cruce = _sentimiento_por_topico_destino(df)

    # Exportar
    df_por_topico.to_csv(
        output_dir / 'sentimiento_por_topico.csv',
        index=False, encoding='utf-8-sig',
    )
    df_por_destino.to_csv(
        output_dir / 'sentimiento_por_destino.csv',
        index=False, encoding='utf-8-sig',
    )
    df_cruce.to_csv(
        output_dir / 'sentimiento_por_topico_destino.csv',
        index=False, encoding='utf-8-sig',
    )

    logger.info('=== Análisis de sentimiento completado. Archivos en: %s ===', output_dir)

    return {
        'corpus'              : df,
        'por_topico'          : df_por_topico,
        'por_destino'         : df_por_destino,
        'por_topico_destino'  : df_cruce,
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_sentiment_analysis()
