'''
hierarchical_clustering.py
--------------------------
Bloque 6 — Clustering Jerárquico (AgglomerativeClustering):
grid search sobre k y método de enlace (linkage).

Score: silhouette directo (único criterio disponible sin inercia).

Funciones públicas:
    evaluar_jerarquico(X, k_rango, metodos) -> list[dict]
'''

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Hiperparámetros del grid search
K_RANGO_DEFAULT  = range(2, 11)
METODOS_DEFAULT  = ['ward', 'complete', 'average', 'single']


def evaluar_jerarquico(
    X: np.ndarray,
    k_rango: range  = K_RANGO_DEFAULT,
    metodos: list   = METODOS_DEFAULT,
) -> list[dict]:
    '''
    Grid search sobre (método de enlace, k) para Clustering Jerárquico.
    Devuelve lista de dicts con métricas por combinación, lista para
    consolidar en el orquestador.

    Cada dict contiene:
        modelo, score_ranking, silhouette, n_clusters,
        n_ruido, hiperparametros, codo_k, _etiquetas (list[int])

    Parámetros:
        X       -- matriz reducida (n_docs x n_dims)
        k_rango -- rango de valores de k a evaluar
        metodos -- lista de métodos de enlace a probar
    '''
    logger.info('Jerárquico: grid search k=%s, métodos=%s', list(k_rango), metodos)

    filas = []

    for metodo in metodos:
        for k in k_rango:
            modelo = AgglomerativeClustering(n_clusters=k, linkage=metodo)
            etiq   = modelo.fit_predict(X)

            # Descarta combinaciones con un solo cluster efectivo
            if len(set(etiq)) < 2:
                logger.debug('Jerárquico k=%d metodo=%s: menos de 2 clusters, saltando', k, metodo)
                continue

            sil = silhouette_score(X, etiq)
            logger.debug('Jerárquico k=%d metodo=%s | silhouette=%.4f', k, metodo, sil)

            filas.append({
                'modelo'         : 'jerarquico',
                'score_ranking'  : round(sil, 6),
                'silhouette'     : round(sil, 6),
                'inercia'        : None,
                'n_clusters'     : k,
                'n_ruido'        : 0,
                'hiperparametros': f'k={k},metodo={metodo}',
                'codo_k'         : None,
                '_etiquetas'     : etiq.tolist(),
            })

    if filas:
        mejor = max(filas, key=lambda r: r['score_ranking'])
        logger.info('Jerárquico: mejor %s | score=%.4f | silhouette=%.4f',
                    mejor['hiperparametros'], mejor['score_ranking'], mejor['silhouette'])
    else:
        logger.warning('Jerárquico: no se generaron resultados válidos')

    return filas