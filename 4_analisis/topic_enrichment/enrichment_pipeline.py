'''
enrichment_pipeline.py
----------------------
Bloque 7 — Orquestador: ejecuta el pipeline completo de enriquecimiento
de tópicos sobre los clusters producidos por el bloque 6.

Para cada fuente disponible en data/clustering/ (embeddings, features,
tfidf, yake) el pipeline ejecuta en orden:

    1. Carga de etiquetas, corpus y matrices vectoriales del bloque 6
    2. Extracción de top keywords por cluster (keyword_extractor.py)
    3. Identificación de documentos representativos (representative_docs.py)
    4. Nombrado automático de tópicos via LLM (topic_naming.py) -- PENDIENTE
    5. Extracción de jerarquía de clusters (topic_hierarchy.py)
    6. Exportación de todos los resultados a data/topic_enrichment/{fuente}/

Estructura de salida en data/topic_enrichment/:
    embeddings/
        keywords_por_cluster.csv
        documentos_representativos.csv
        jerarquia.csv
        topic_names.json           se generará cuando topic_naming esté implementado
    features/
        (mismos archivos)
    tfidf/
        (mismos archivos)
    yake/
        (mismos archivos)
    resumen_enrichment.csv         tabla consolidada con métricas por fuente y cluster

Uso desde main.py:
    from topic_enrichment.enrichment_pipeline import run_enrichment_pipeline
    run_enrichment_pipeline()
'''

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from topic_enrichment.keyword_extractor import (
    extract_top_keywords,
    keywords_to_dataframe,
)
from topic_enrichment.representative_docs import (
    get_representative_docs,
    representative_docs_to_dataframe,
)
from topic_enrichment.topic_hierarchy import build_full_hierarchy, hierarchy_to_dataframe

# topic_naming se importará aquí cuando esté implementado:
# from topic_enrichment.topic_naming import name_all_clusters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ======================================================
# RUTAS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Fuentes producidas por el bloque 6
DIR_CLUSTERING = DATA_DIR / 'clustering'

# Corpus limpio — se usa en todos los enriquecimientos
PATH_CLEAN_CSV = DATA_DIR / 'data_spanish' / 'clean.csv'
COLUMNA_TEXTO  = 'comentario_clean'

# Salida del bloque 7
DIR_ENRICHMENT = DATA_DIR / 'topic_enrichment'

# Fuentes disponibles — mismo orden que en clustering_pipeline.py
FUENTES_DISPONIBLES = ['embeddings', 'features', 'tfidf', 'yake']

# Hiperparámetros del enrichment
TOP_N_KEYWORDS = 15   # keywords a extraer por cluster
TOP_K_DOCS     = 5    # documentos representativos por cluster
MIN_FREQ_VOCAB = 2    # frecuencia mínima para incluir un término en el vocabulario


# ======================================================
# CARGA DE DATOS
# ======================================================

def _cargar_corpus(path: Path) -> list[str]:
    '''
    Lee el corpus limpio desde clean.csv y retorna la lista de textos.
    Documentos nulos o vacíos se reemplazan con cadena vacía.
    '''
    df = pd.read_csv(path)
    textos = df[COLUMNA_TEXTO].fillna('').astype(str).tolist()
    logger.info('Corpus cargado: %d documentos desde %s', len(textos), path)
    return textos


def _cargar_etiquetas_fuente(dir_fuente: Path) -> dict[str, list[int]] | None:
    '''
    Carga el archivo etiquetas_mejores.json de una fuente del bloque 6.
    Retorna None si el archivo no existe o tiene errores.

    El JSON contiene {clave_modelo: [etiquetas por documento]}.
    '''
    path_etiq = dir_fuente / 'etiquetas_mejores.json'

    if not path_etiq.exists():
        logger.warning('etiquetas_mejores.json no encontrado en %s', dir_fuente)
        return None

    with open(path_etiq, encoding='utf-8') as f:
        etiquetas = json.load(f)

    logger.info(
        'Etiquetas cargadas desde %s: %d modelos',
        dir_fuente.name, len(etiquetas)
    )
    return etiquetas


def _cargar_proyeccion_2d(dir_fuente: Path) -> np.ndarray | None:
    '''
    Carga la proyección 2D guardada por clustering_pipeline.py.
    Se usa como espacio vectorial para calcular centroides y similitudes.
    Retorna None si el archivo no existe.
    '''
    path_npy = dir_fuente / 'proyeccion_2d.npy'

    if not path_npy.exists():
        logger.warning(
            'proyeccion_2d.npy no encontrado en %s — '
            'documentos representativos se calcularán sin este espacio',
            dir_fuente
        )
        return None

    X = np.load(path_npy)
    logger.info('Proyección 2D cargada: shape=%s desde %s', X.shape, path_npy)
    return X


# ======================================================
# EXPORTACIÓN
# ======================================================

def _exportar_resultados_fuente(
    dir_salida: Path,
    df_keywords: pd.DataFrame,
    df_docs: pd.DataFrame,
    df_jerarquia: pd.DataFrame,
    nombre_modelo: str,
) -> None:
    '''
    Exporta los tres DataFrames de enriquecimiento a CSV dentro de la
    subcarpeta de la fuente y modelo correspondiente.

    Se crea una subcarpeta por modelo ({fuente}/{nombre_modelo}/) para
    separar los resultados cuando una fuente tiene múltiples modelos
    (ej: embeddings con jerarquico y kmeans).
    '''
    dir_modelo = dir_salida / nombre_modelo.replace('|', '_').replace('=', '')
    dir_modelo.mkdir(parents=True, exist_ok=True)

    path_kw  = dir_modelo / 'keywords_por_cluster.csv'
    path_docs = dir_modelo / 'documentos_representativos.csv'
    path_jer  = dir_modelo / 'jerarquia.csv'

    df_keywords.to_csv(path_kw,   index=False, encoding='utf-8-sig')
    df_docs.to_csv(path_docs,     index=False, encoding='utf-8-sig')
    df_jerarquia.to_csv(path_jer, index=False, encoding='utf-8-sig')

    logger.info(
        'Resultados exportados en %s: '
        '%d keywords, %d docs representativos, %d nodos jerarquía',
        dir_modelo, len(df_keywords), len(df_docs), len(df_jerarquia)
    )


def _exportar_resumen_global(
    filas_resumen: list[dict],
) -> None:
    '''
    Exporta la tabla resumen consolidada con métricas de enriquecimiento
    por fuente, modelo y cluster al archivo resumen_enrichment.csv.
    '''
    if not filas_resumen:
        logger.warning('Sin datos para exportar el resumen global de enrichment')
        return

    DIR_ENRICHMENT.mkdir(parents=True, exist_ok=True)
    path_resumen = DIR_ENRICHMENT / 'resumen_enrichment.csv'

    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(path_resumen, index=False, encoding='utf-8-sig')

    logger.info(
        'Resumen global exportado: %d filas en %s',
        len(df_resumen), path_resumen
    )


# ======================================================
# ENRICHMENT POR FUENTE Y MODELO
# ======================================================

def _enriquecer_modelo(
    nombre_fuente: str,
    nombre_modelo: str,
    labels: list[int],
    corpus: list[str],
    X: np.ndarray | None,
    dir_salida: Path,
    filas_resumen: list[dict],
) -> None:
    '''
    Ejecuta el pipeline completo de enriquecimiento para un modelo específico
    dentro de una fuente.

    Parámetros:
        nombre_fuente  -- nombre de la fuente (embeddings, features, tfidf, yake)
        nombre_modelo  -- clave del modelo (ej: 'jerarquico|k=5,metodo=ward')
        labels         -- etiquetas de cluster por documento
        corpus         -- lista de textos limpios
        X              -- matriz de vectores 2D (puede ser None)
        dir_salida     -- directorio de salida para esta fuente
        filas_resumen  -- lista acumuladora para el resumen global
    '''
    labels_array = np.array(labels)
    cluster_ids = sorted(set(labels_array))
    cluster_ids_validos = [c for c in cluster_ids if c != -1]
    n_clusters = len(cluster_ids_validos)

    logger.info(
        '[%s | %s] Enriqueciendo %d clusters (%d documentos total)',
        nombre_fuente, nombre_modelo, n_clusters, len(labels)
    )

    # --- Paso 1: Top keywords por cluster ---
    logger.info('[%s | %s] Extrayendo keywords...', nombre_fuente, nombre_modelo)
    keywords_por_cluster = extract_top_keywords(
        corpus=corpus,
        labels=labels_array,
        top_n=TOP_N_KEYWORDS,
        min_freq=MIN_FREQ_VOCAB,
    )
    df_keywords = keywords_to_dataframe(keywords_por_cluster)

    # --- Paso 2: Documentos representativos ---
    logger.info('[%s | %s] Identificando documentos representativos...', nombre_fuente, nombre_modelo)
    if X is not None:
        docs_por_cluster = get_representative_docs(
            X=X,
            labels=labels_array,
            corpus=corpus,
            top_k=TOP_K_DOCS,
        )
        df_docs = representative_docs_to_dataframe(docs_por_cluster)
    else:
        logger.warning(
            '[%s | %s] Sin matriz vectorial — documentos representativos omitidos',
            nombre_fuente, nombre_modelo
        )
        df_docs = pd.DataFrame()

    # --- Paso 3: Topic naming --- PENDIENTE
    # Cuando topic_naming.py esté implementado, se activará aquí:
    # topic_names = name_all_clusters(keywords_por_cluster, docs_por_cluster)
    logger.info(
        '[%s | %s] Topic naming pendiente de implementación — omitido',
        nombre_fuente, nombre_modelo
    )

    # --- Paso 4: Jerarquía ---
    logger.info('[%s | %s] Extrayendo jerarquía...', nombre_fuente, nombre_modelo)
    if X is not None and n_clusters >= 2:
        try:
            _, df_jerarquia = build_full_hierarchy(X, labels_array, method='ward')
        except Exception as error:
            logger.warning(
                '[%s | %s] Error al construir jerarquía: %s — omitida',
                nombre_fuente, nombre_modelo, error
            )
            df_jerarquia = pd.DataFrame()
    else:
        logger.warning(
            '[%s | %s] Jerarquía omitida (sin matriz vectorial o menos de 2 clusters)',
            nombre_fuente, nombre_modelo
        )
        df_jerarquia = pd.DataFrame()

    # --- Exportar resultados ---
    _exportar_resultados_fuente(
        dir_salida, df_keywords, df_docs, df_jerarquia, nombre_modelo
    )

    # --- Acumular resumen ---
    for cluster_id in cluster_ids_validos:
        n_docs_cluster = int((labels_array == cluster_id).sum())
        top_keyword = (
            df_keywords[df_keywords['cluster_id'] == cluster_id]['termino'].iloc[0]
            if not df_keywords.empty and cluster_id in df_keywords['cluster_id'].values
            else 'N/A'
        )
        filas_resumen.append({
            'fuente'         : nombre_fuente,
            'modelo'         : nombre_modelo,
            'cluster_id'     : cluster_id,
            'n_docs_cluster' : n_docs_cluster,
            'top_keyword'    : top_keyword,
            'n_keywords'     : len(keywords_por_cluster.get(cluster_id, [])),
            'n_docs_repr'    : len(docs_por_cluster.get(cluster_id, [])) if X is not None else 0,
        })


def _enriquecer_fuente(
    nombre_fuente: str,
    corpus: list[str],
    filas_resumen: list[dict],
) -> None:
    '''
    Orquesta el enriquecimiento de todos los modelos de una fuente.
    Carga etiquetas y proyección 2D, luego llama a _enriquecer_modelo
    para cada modelo disponible.
    '''
    dir_fuente = DIR_CLUSTERING / nombre_fuente

    if not dir_fuente.exists():
        logger.info('Subcarpeta %s no existe — fuente omitida', dir_fuente)
        return

    etiquetas_por_modelo = _cargar_etiquetas_fuente(dir_fuente)
    if etiquetas_por_modelo is None:
        return

    X = _cargar_proyeccion_2d(dir_fuente)

    dir_salida = DIR_ENRICHMENT / nombre_fuente
    dir_salida.mkdir(parents=True, exist_ok=True)

    for nombre_modelo, labels in etiquetas_por_modelo.items():
        logger.info('=== [%s] Modelo: %s ===', nombre_fuente, nombre_modelo)

        # Verificar alineación entre corpus y etiquetas
        if len(labels) != len(corpus):
            logger.error(
                '[%s | %s] Etiquetas (%d) no alineadas con corpus (%d) — modelo omitido',
                nombre_fuente, nombre_modelo, len(labels), len(corpus)
            )
            continue

        _enriquecer_modelo(
            nombre_fuente=nombre_fuente,
            nombre_modelo=nombre_modelo,
            labels=labels,
            corpus=corpus,
            X=X,
            dir_salida=dir_salida,
            filas_resumen=filas_resumen,
        )

    logger.info('Fuente %s completada', nombre_fuente)


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def run_enrichment_pipeline(
    fuentes: list[str] | None = None,
) -> None:
    '''
    Pipeline completo de enriquecimiento de tópicos.

    Itera sobre las fuentes disponibles en data/clustering/ y ejecuta
    el enriquecimiento completo (keywords, docs representativos, jerarquía)
    para cada modelo dentro de cada fuente.

    Parámetros:
        fuentes -- lista de fuentes a procesar; si es None se procesan
                   todas las disponibles (embeddings, features, tfidf, yake)
    '''
    fuentes_a_procesar = fuentes if fuentes is not None else FUENTES_DISPONIBLES

    logger.info('Iniciando pipeline de topic enrichment')
    logger.info('Fuentes a procesar: %s', fuentes_a_procesar)

    # Cargar corpus una sola vez — se reutiliza para todas las fuentes
    if not PATH_CLEAN_CSV.exists():
        logger.error(
            'Corpus no encontrado en %s — pipeline abortado', PATH_CLEAN_CSV
        )
        return

    corpus = _cargar_corpus(PATH_CLEAN_CSV)

    # Lista acumuladora para el resumen global
    filas_resumen: list[dict] = []

    # Procesar cada fuente
    for nombre_fuente in fuentes_a_procesar:
        logger.info('=== FUENTE: %s ===', nombre_fuente)
        _enriquecer_fuente(nombre_fuente, corpus, filas_resumen)

    # Exportar resumen consolidado
    _exportar_resumen_global(filas_resumen)

    logger.info('Pipeline de topic enrichment completado')


if __name__ == '__main__':
    run_enrichment_pipeline()