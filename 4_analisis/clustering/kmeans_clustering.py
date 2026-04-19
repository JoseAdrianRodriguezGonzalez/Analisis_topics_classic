'''
kmeans_clustering.py
--------------------
Bloque 6 — KMeans: grid search sobre el número de clusters k.

Score combinado: alpha * silhouette_norm + (1 - alpha) * elbow_score
El codo se detecta como la k de máxima curvatura (segunda derivada) de la
curva de inercias.

Funciones públicas:
    evaluar_kmeans(X, k_rango, alpha) -> list[dict]
'''

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Hiperparámetros del grid search
K_RANGO_DEFAULT  = range(2, 11)
ALPHA_DEFAULT    = 0.7   # peso silhouette vs elbow (0-1)


def _detectar_codo(inercias: dict, ks: list[int]) -> int:
    '''
    Detecta la k óptima como el punto de máxima curvatura de la curva
    de inercias usando la segunda derivada discreta.
    '''
    iv  = np.array([inercias[k] for k in ks])
    d2  = np.diff(np.diff(iv))
    # +1 porque diff reduce longitud en 1 dos veces
    return ks[int(np.argmax(np.abs(d2))) + 1]


def evaluar_kmeans(
    X: np.ndarray,
    k_rango: range = K_RANGO_DEFAULT,
    alpha: float   = ALPHA_DEFAULT,
) -> list[dict]:
    '''
    Grid search sobre k para KMeans. Devuelve lista de dicts con métricas
    por combinación de hiperparámetros, lista para consolidar en el orquestador.

    Cada dict contiene:
        modelo, score_ranking, silhouette, inercia, n_clusters,
        n_ruido, hiperparametros, codo_k, _etiquetas (list[int])

    Parámetros:
        X       -- matriz reducida (n_docs x n_dims)
        k_rango -- rango de valores de k a evaluar
        alpha   -- peso de silhouette en el score combinado
    '''
    logger.info('KMeans: grid search k=%s, alpha=%.2f', list(k_rango), alpha)

    inercias    : dict[int, float]      = {}
    silhouettes : dict[int, float]      = {}
    etiquetas   : dict[int, np.ndarray] = {}

    for k in k_rango:
        modelo        = KMeans(n_clusters=k, random_state=42, n_init='auto')
        etiq          = modelo.fit_predict(X)
        inercias[k]   = modelo.inertia_
        silhouettes[k] = silhouette_score(X, etiq)
        etiquetas[k]  = etiq
        logger.debug('KMeans k=%d | silhouette=%.4f | inercia=%.2f', k, silhouettes[k], inercias[k])

    ks     = list(k_rango)
    codo_k = _detectar_codo(inercias, ks)
    logger.info('KMeans: codo detectado en k=%d', codo_k)

    # Elbow score: decae linealmente con la distancia al codo
    elbow_sc = {k: max(0.0, 1.0 - abs(k - codo_k) / len(ks)) for k in ks}

    # Normalizar silhouette a [0, 1] dentro del grid
    s_arr  = np.array([silhouettes[k] for k in ks])
    s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-9)

    filas = []
    for i, k in enumerate(ks):
        score = alpha * s_norm[i] + (1.0 - alpha) * elbow_sc[k]
        filas.append({
            'modelo'         : 'kmeans',
            'score_ranking'  : round(score, 6),
            'silhouette'     : round(silhouettes[k], 6),
            'inercia'        : round(inercias[k], 4),
            'n_clusters'     : k,
            'n_ruido'        : 0,
            'hiperparametros': f'k={k}',
            'codo_k'         : codo_k,
            '_etiquetas'     : etiquetas[k].tolist(),
        })

    mejor = max(filas, key=lambda r: r['score_ranking'])
    logger.info('KMeans: mejor k=%d | score=%.4f | silhouette=%.4f',
                mejor['n_clusters'], mejor['score_ranking'], mejor['silhouette'])
    return filas