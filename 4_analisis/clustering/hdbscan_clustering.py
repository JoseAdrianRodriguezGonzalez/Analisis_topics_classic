'''
hdbscan_clustering.py
---------------------
Bloque 6 — HDBSCAN: grid search sobre min_cluster_size y min_samples.

Score: silhouette * (1 - pct_ruido) * penalizacion_balance
    - pct_ruido      : fracción de puntos marcados como ruido (-1)
    - penalizacion   : 0 si algún cluster acapara > MAX_CLUSTER_PCT de puntos

Funciones públicas:
    evaluar_hdbscan(X, min_cluster_sizes, min_samples_list) -> list[dict]
'''

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Hiperparámetros del grid search
MIN_CLUSTER_SIZES_DEFAULT = [5, 10, 15, 20, 30]
MIN_SAMPLES_DEFAULT       = [3, 5, 10]

# Umbral de penalización: un cluster no puede concentrar más del 80% de los puntos
MAX_CLUSTER_PCT = 0.80


def evaluar_hdbscan(
    X: np.ndarray,
    min_cluster_sizes: list[int] = MIN_CLUSTER_SIZES_DEFAULT,
    min_samples_list : list[int] = MIN_SAMPLES_DEFAULT,
) -> list[dict]:
    '''
    Grid search sobre (min_cluster_size, min_samples) para HDBSCAN.
    Devuelve lista de dicts con métricas por combinación, lista para
    consolidar en el orquestador.

    Requiere que el paquete hdbscan esté instalado:
        pip install hdbscan

    Cada dict contiene:
        modelo, score_ranking, silhouette, n_clusters,
        n_ruido, hiperparametros, codo_k, _etiquetas (list[int])

    Parámetros:
        X                  -- matriz reducida (n_docs x n_dims)
        min_cluster_sizes  -- lista de valores de min_cluster_size a probar
        min_samples_list   -- lista de valores de min_samples a probar
    '''
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        logger.error('hdbscan no está instalado. Ejecuta: pip install hdbscan')
        return []

    logger.info('HDBSCAN: grid search min_cluster_size=%s, min_samples=%s',
                min_cluster_sizes, min_samples_list)

    n_total              = X.shape[0]
    # Tamaño mínimo de cluster válido: al menos 5% del corpus
    umbral_cluster_minimo = max(3, int(n_total * 0.05))
    filas                = []

    for min_cs in min_cluster_sizes:
        for min_s in min_samples_list:
            modelo = HDBSCAN(min_cluster_size=min_cs, min_samples=min_s)
            etiq   = modelo.fit_predict(X)

            mascara    = etiq != -1
            n_ruido    = int((~mascara).sum())
            n_validos  = int(mascara.sum())
            n_clusters = len(set(etiq[mascara])) if n_validos > 0 else 0

            # Descarta soluciones triviales
            if n_clusters < 2 or n_validos < 4:
                logger.debug('HDBSCAN min_cs=%d min_s=%d: clusters=%d, saltando',
                             min_cs, min_s, n_clusters)
                continue

            counts = np.bincount(etiq[mascara])

            # Descarta si todos los clusters son demasiado pequeños
            if counts.max() <= umbral_cluster_minimo:
                continue

            sil      = silhouette_score_safe(X[mascara], etiq[mascara])
            if sil is None:
                continue

            pct_ruido = n_ruido / n_total
            pct_max   = counts.max() / n_validos
            penalizacion = 0.0 if pct_max > MAX_CLUSTER_PCT else 1.0
            score     = sil * (1.0 - pct_ruido) * penalizacion

            if score <= 0:
                continue

            logger.debug('HDBSCAN min_cs=%d min_s=%d | clusters=%d ruido=%d sil=%.4f score=%.4f',
                         min_cs, min_s, n_clusters, n_ruido, sil, score)

            filas.append({
                'modelo'         : 'hdbscan',
                'score_ranking'  : round(score, 6),
                'silhouette'     : round(sil, 6),
                'inercia'        : None,
                'n_clusters'     : n_clusters,
                'n_ruido'        : n_ruido,
                'hiperparametros': f'min_cluster_size={min_cs},min_samples={min_s}',
                'codo_k'         : None,
                '_etiquetas'     : etiq.tolist(),
            })

    if filas:
        mejor = max(filas, key=lambda r: r['score_ranking'])
        logger.info('HDBSCAN: mejor %s | score=%.4f | silhouette=%.4f | clusters=%d | ruido=%d',
                    mejor['hiperparametros'], mejor['score_ranking'],
                    mejor['silhouette'], mejor['n_clusters'], mejor['n_ruido'])
    else:
        logger.warning('HDBSCAN: ninguna combinación produjo resultados válidos')

    return filas


def silhouette_score_safe(X: np.ndarray, etiq: np.ndarray) -> float | None:
    '''
    Calcula silhouette_score con manejo de errores.
    Retorna None si el cálculo falla (p. ej. todos los puntos en un cluster).
    '''
    from sklearn.metrics import silhouette_score
    try:
        return silhouette_score(X, etiq)
    except ValueError as e:
        logger.warning('silhouette_score falló: %s', e)
        return None