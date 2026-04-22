'''
analysis_pipeline.py
--------------------
Bloque 8 — Orquestador principal del análisis semántico y estadístico.

Ejecuta los cuatro módulos de análisis en el orden correcto y
exporta un resumen consolidado de los resultados.

Orden de ejecución:
    1. sentiment_analysis   -- sentimiento basado en estrellas + POS
    2. entity_analysis      -- cruza NER con sentimiento (depende de 1)
    3. cooccurrence_graph   -- co-ocurrencia entre entidades y términos
    4. trend_detection      -- distribución temática y perfiles (depende de 1)

Estructura de salida en data/analysis/:
    sentiment/
        corpus_con_sentimiento.csv
        sentimiento_por_topico.csv
        sentimiento_por_destino.csv
        sentimiento_por_topico_destino.csv
    entities/
        entidades_con_sentimiento.csv
        entidades_por_destino.csv
        entidades_por_topico.csv
    cooccurrence/
        coocurrencia_entidades.csv
        coocurrencia_terminos.csv
        comunidades_entidades.csv
    trends/
        tendencias_topicos_destino.csv
        tendencias_sentimiento_topico.csv
        microtopicos_resumen.csv
        perfil_destino.csv
    resumen_analysis.csv   <-- tabla consolidada con métricas clave

Uso desde main.py:
    from analysis.analysis_pipeline import run_analysis_pipeline
    run_analysis_pipeline()

Uso directo:
    python analysis_pipeline.py
'''

import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ======================================================
# RUTAS
# ======================================================

BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / 'data'
OUTPUT_DIR   = DATA_DIR / 'analysis'


# ======================================================
# RESUMEN CONSOLIDADO
# ======================================================

def _generar_resumen_consolidado(
    resultados: dict[str, dict],
    output_dir: Path,
) -> pd.DataFrame:
    '''
    Genera una tabla resumen con las métricas más importantes de cada
    módulo para facilitar la interpretación global del análisis.

    La tabla tiene una fila por destino con columnas de cada módulo.
    '''
    filas = []

    # Métricas de sentimiento por destino
    df_sent = resultados.get('sentiment', {}).get('por_destino')
    if df_sent is not None and not df_sent.empty:
        for _, fila in df_sent.iterrows():
            registro = {
                'location'           : fila.get('location', ''),
                'n_total_docs'       : fila.get('total_documentos', 0),
                'n_con_rating'       : fila.get('total_con_rating', 0),
                'sentimiento_medio'  : fila.get('sentimiento_medio', None),
                'estrella_media'     : fila.get('estrella_media', None),
                'pct_positivo'       : fila.get('pct_positivo', None),
                'pct_negativo'       : fila.get('pct_negativo', None),
            }
            filas.append(registro)

    # Enriquecer con datos de tendencias (perfil de destino)
    df_perfil = resultados.get('trends', {}).get('perfil_destino')
    if df_perfil is not None and not df_perfil.empty and filas:
        df_resumen = pd.DataFrame(filas)
        cols_merge = [c for c in ['topico_dominante', 'n_topicos_relevantes'] if c in df_perfil.columns]
        if cols_merge:
            df_resumen = df_resumen.merge(
                df_perfil[['location'] + cols_merge],
                on='location',
                how='left',
            )
        filas = df_resumen.to_dict('records')

    df_resumen = pd.DataFrame(filas)

    if not df_resumen.empty:
        path_resumen = output_dir / 'resumen_analysis.csv'
        df_resumen.to_csv(path_resumen, index=False, encoding='utf-8-sig')
        logger.info('Resumen consolidado exportado: %s', path_resumen)

    return df_resumen


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def run_analysis_pipeline(
    ejecutar_sentiment    : bool = True,
    ejecutar_entities     : bool = True,
    ejecutar_cooccurrence : bool = True,
    ejecutar_trends       : bool = True,
) -> dict[str, dict]:
    '''
    Orquestador principal del bloque 8 de análisis.

    Ejecuta en orden los cuatro módulos de análisis y genera un
    resumen consolidado. Cada módulo puede activarse o desactivarse
    de forma independiente.

    Parámetros:
        ejecutar_sentiment    -- si True, ejecuta análisis de sentimiento
        ejecutar_entities     -- si True, ejecuta análisis de entidades
        ejecutar_cooccurrence -- si True, construye grafo de co-ocurrencia
        ejecutar_trends       -- si True, ejecuta detección de tendencias

    Retorna diccionario anidado:
        {
            'sentiment'    : {corpus, por_topico, por_destino, ...},
            'entities'     : {entidades, por_destino, por_topico},
            'cooccurrence' : {cooc_entidades, cooc_terminos, comunidades},
            'trends'       : {dist_topicos_destino, perfil_sentimiento_topico, ...},
        }
    '''
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resultados: dict[str, dict] = {}
    inicio_total = time.time()

    logger.info('=== BLOQUE 8: PIPELINE DE ANÁLISIS — INICIO ===')

    # --------------------------------------------------
    # Paso 1: Análisis de sentimiento
    # --------------------------------------------------
    if ejecutar_sentiment:
        logger.info('\n--- Paso 1/4: Análisis de sentimiento ---')
        t0 = time.time()
        try:
            from analysis.sentiment_analysis import run_sentiment_analysis
            resultados['sentiment'] = run_sentiment_analysis()
            logger.info('Sentimiento completado en %.1f s', time.time() - t0)
        except Exception as error:
            logger.error('Error en análisis de sentimiento: %s', error)
            resultados['sentiment'] = {}
    else:
        logger.info('Análisis de sentimiento omitido por configuración')

    # --------------------------------------------------
    # Paso 2: Análisis de entidades NER
    # --------------------------------------------------
    if ejecutar_entities:
        logger.info('\n--- Paso 2/4: Análisis de entidades NER ---')
        t0 = time.time()
        try:
            from analysis.entity_analysis import run_entity_analysis
            resultados['entities'] = run_entity_analysis()
            logger.info('Entidades completado en %.1f s', time.time() - t0)
        except Exception as error:
            logger.error('Error en análisis de entidades: %s', error)
            resultados['entities'] = {}
    else:
        logger.info('Análisis de entidades omitido por configuración')

    # --------------------------------------------------
    # Paso 3: Grafo de co-ocurrencia
    # --------------------------------------------------
    if ejecutar_cooccurrence:
        logger.info('\n--- Paso 3/4: Grafo de co-ocurrencia ---')
        t0 = time.time()
        try:
            from analysis.cooccurrence_graph import run_cooccurrence_graph
            resultados['cooccurrence'] = run_cooccurrence_graph()
            logger.info('Co-ocurrencia completada en %.1f s', time.time() - t0)
        except Exception as error:
            logger.error('Error en co-ocurrencia: %s', error)
            resultados['cooccurrence'] = {}
    else:
        logger.info('Grafo de co-ocurrencia omitido por configuración')

    # --------------------------------------------------
    # Paso 4: Detección de tendencias
    # --------------------------------------------------
    if ejecutar_trends:
        logger.info('\n--- Paso 4/4: Detección de tendencias ---')
        t0 = time.time()
        try:
            from analysis.trend_detection import run_trend_detection
            resultados['trends'] = run_trend_detection()
            logger.info('Tendencias completadas en %.1f s', time.time() - t0)
        except Exception as error:
            logger.error('Error en detección de tendencias: %s', error)
            resultados['trends'] = {}
    else:
        logger.info('Detección de tendencias omitida por configuración')

    # --------------------------------------------------
    # Resumen consolidado
    # --------------------------------------------------
    if resultados:
        logger.info('\n--- Generando resumen consolidado ---')
        _generar_resumen_consolidado(resultados, OUTPUT_DIR)

    duracion_total = time.time() - inicio_total
    logger.info(
        '\n=== BLOQUE 8 COMPLETADO en %.1f s ===',
        duracion_total,
    )
    logger.info('Archivos en: %s', OUTPUT_DIR)

    # Imprimir resumen de archivos generados
    archivos_generados = list(OUTPUT_DIR.rglob('*.csv'))
    logger.info('\nArchivos generados: %d', len(archivos_generados))
    for archivo in sorted(archivos_generados):
        ruta_relativa = archivo.relative_to(OUTPUT_DIR)
        logger.info('  %s', ruta_relativa)

    return resultados


if __name__ == '__main__':
    run_analysis_pipeline()