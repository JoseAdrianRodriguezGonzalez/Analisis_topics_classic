'''
Clustering.py
-------------
Evaluación exhaustiva de K-means, Jerárquico y DBSCAN
sobre matrices TF-IDF reducidas con PCA, UMAP y t-SNE.

Salidas en  data/clusterizacion/
  ranking_completo.csv   - todas las combinaciones ordenadas
  mejores_modelos.csv    - top-1 por (ngrama, modelo)
  etiquetas_mejores.json - etiquetas para GraficadoClusterizacion.py
  PCA/ UMAP/ TSNE/       - carpetas para las graficas
'''

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap import UMAP

warnings.filterwarnings('ignore')

# ======================================================
# CONFIGURACION
# ======================================================

DIR_DATOS  = '../data/processed/'
DIR_BASE   = '../data/clusterizacion/'
PATH_NLP   = DIR_DATOS + 'data_nlp.csv'

ARCHIVOS_TFIDF = {
    'unigramas': DIR_DATOS + 'TF_IDF_normalizado_unigramas.csv',
    'bigramas' : DIR_DATOS + 'TF_IDF_normalizado_bigramas.csv',
    'trigramas': DIR_DATOS + 'TF_IDF_normalizado_trigramas.csv',
}

K_RANGO          = range(2, 11)
METODOS_JER      = ['ward', 'complete', 'average', 'single']
EPSILONS         = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
MIN_SAMPLES_LIST = list(range(2, 8))

# Peso silhouette vs componente elbow en el score combinado de K-means
ALPHA_KMEANS = 0.7

# Umbral: un cluster no puede contener más del 80 % de los puntos no-ruido
# (para DBSCAN y como referencia en K-means)
MAX_CLUSTER_PCT = 0.80

for red in ['PCA', 'UMAP', 'TSNE']:
    os.makedirs(os.path.join(DIR_BASE, red), exist_ok=True)


# ======================================================
# CARGA DE DATOS
# ======================================================

def cargar_matriz(path):
    df   = pd.read_csv(path)
    mask = df.values.sum(axis=1) != 0
    return df.values[mask].astype(float), np.where(mask)[0]


def cargar_metadatos(path, indices_validos):
    return pd.read_csv(path).iloc[indices_validos].reset_index(drop=True)


# ======================================================
# REDUCCION DE DIMENSIONALIDAD
# ======================================================

def reducir(X, metodo):
    if metodo == 'PCA':
        return PCA(n_components=2, random_state=42).fit_transform(X)
    if metodo == 'UMAP':
        return UMAP(n_components=2, random_state=42).fit_transform(X)
    if metodo == 'TSNE':
        perp = min(30, X.shape[0] - 1)
        return TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)


# ======================================================
# EVALUACION K-MEANS
# Score combinado = α·silhouette_norm + (1-α)·elbow_score
# ======================================================

def evaluar_kmeans(X, ngrama, reduccion):
    inercias    = {}
    silhouettes = {}
    etiquetas   = {}

    for k in K_RANGO:
        m            = KMeans(n_clusters=k, random_state=42, n_init='auto')
        etiq         = m.fit_predict(X)
        inercias[k]  = m.inertia_
        silhouettes[k] = silhouette_score(X, etiq)
        etiquetas[k]   = etiq

    # Componente elbow: maxima curvatura de la curva de inercias
    ks     = list(K_RANGO)
    iv     = np.array([inercias[k] for k in ks])
    d2     = np.diff(np.diff(iv))                      # segunda derivada
    codo_k = ks[int(np.argmax(np.abs(d2))) + 1]        # +1 por doble diff

    # Elbow score: decae linealmente con la distancia al codo
    elbow_sc = {k: max(0.0, 1.0 - abs(k - codo_k) / len(ks)) for k in ks}

    # Normalizar silhouette a [0,1] dentro de este conjunto
    s_arr  = np.array([silhouettes[k] for k in ks])
    s_min, s_max = s_arr.min(), s_arr.max()
    s_norm = (s_arr - s_min) / (s_max - s_min + 1e-9)

    filas = []
    for i, k in enumerate(ks):
        score = ALPHA_KMEANS * s_norm[i] + (1 - ALPHA_KMEANS) * elbow_sc[k]
        filas.append({
            'ngrama'         : ngrama,
            'modelo'         : 'kmeans',
            'reduccion'      : reduccion,
            'score_ranking'  : round(score, 6),
            'silhouette'     : round(silhouettes[k], 6),
            'inercia'        : round(inercias[k], 4),
            'codo_k'         : codo_k,
            'n_clusters'     : k,
            'n_ruido'        : 0,
            'pct_ruido'      : 0.0,
            'hiperparametros': f'k={k}',
            '_etiquetas'     : etiquetas[k].tolist(),
        })
    return filas


# ======================================================
# EVALUACION JERARQUICO
# Score = silhouette (único criterio disponible sin inercia)
# ======================================================

def evaluar_jerarquico(X, ngrama, reduccion):
    filas = []
    for metodo in METODOS_JER:
        for k in K_RANGO:
            m    = AgglomerativeClustering(n_clusters=k, linkage=metodo)
            etiq = m.fit_predict(X)
            if len(set(etiq)) < 2:
                continue
            sil = silhouette_score(X, etiq)
            filas.append({
                'ngrama'         : ngrama,
                'modelo'         : 'jerarquico',
                'reduccion'      : reduccion,
                'score_ranking'  : round(sil, 6),
                'silhouette'     : round(sil, 6),
                'inercia'        : None,
                'codo_k'         : None,
                'n_clusters'     : k,
                'n_ruido'        : 0,
                'pct_ruido'      : 0.0,
                'hiperparametros': f'k={k},metodo={metodo}',
                '_etiquetas'     : etiq.tolist(),
            })
    return filas


# ======================================================
# EVALUACION DBSCAN
# Score = silhouette × (1 - pct_ruido) × penalty_balance
# penalty_balance = 0 si un cluster tiene >80% de los puntos
# ======================================================

def evaluar_dbscan(X, ngrama, reduccion):
    n_total = X.shape[0]
    filas   = []

    # Umbral minimo: al menos un cluster debe tener el 5% del corpus
    MIN_CLUSTER_SIZE = max(3, int(n_total * 0.05))

    for eps in EPSILONS:
        for min_s in MIN_SAMPLES_LIST:
            m    = DBSCAN(eps=eps, min_samples=min_s)
            etiq = m.fit_predict(X)

            mascara    = etiq != -1
            n_ruido    = int(np.sum(~mascara))
            n_validos  = int(mascara.sum())
            n_clusters = len(set(etiq[mascara])) if n_validos > 0 else 0

            if n_clusters < 2 or n_validos < 4:
                continue

            counts = np.bincount(etiq[mascara])

            # Descartar si todos los clusters son trivialmente pequeños
            # evita casos como 35 clusters de 2 docs (overfitting geometrico)
            if counts.max() <= MIN_CLUSTER_SIZE:
                continue

            sil       = silhouette_score(X[mascara], etiq[mascara])
            pct_ruido = n_ruido / n_total

            # Penalizar si un solo cluster acapara más del 80% de los puntos
            pct_max = counts.max() / n_validos
            penalty = 0.0 if pct_max > MAX_CLUSTER_PCT else 1.0

            score = sil * (1.0 - pct_ruido) * penalty
            if score <= 0:
                continue

            filas.append({
                'ngrama'         : ngrama,
                'modelo'         : 'dbscan',
                'reduccion'      : reduccion,
                'score_ranking'  : round(score, 6),
                'silhouette'     : round(sil, 6),
                'inercia'        : None,
                'codo_k'         : None,
                'n_clusters'     : n_clusters,
                'n_ruido'        : n_ruido,
                'pct_ruido'      : round(pct_ruido, 4),
                'hiperparametros': f'eps={eps},min_samples={min_s}',
                '_etiquetas'     : etiq.tolist(),
            })
    return filas


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

todos = []

for ngrama, path in ARCHIVOS_TFIDF.items():
    print(f'\nNgrama: {ngrama}')

    X, indices_validos = cargar_matriz(path)
    meta = cargar_metadatos(PATH_NLP, indices_validos)
    print(f'  Documentos: {X.shape[0]}  |  Vocabulario: {X.shape[1]}')

    for reduccion in ['PCA', 'UMAP', 'TSNE']:
        print(f'  {reduccion}...')
        X_red = reducir(X, reduccion)

        todos += evaluar_kmeans(X_red, ngrama, reduccion)
        todos += evaluar_jerarquico(X_red, ngrama, reduccion)
        todos += evaluar_dbscan(X_red, ngrama, reduccion)


# ======================================================
# RANKING Y EXPORTACION
# ======================================================

etiquetas_dict = {}
filas_csv      = []

for row in todos:
    etiq = row.pop('_etiquetas')
    key  = f"{row['ngrama']}|{row['modelo']}|{row['reduccion']}|{row['hiperparametros']}"
    etiquetas_dict[key] = etiq
    filas_csv.append(row)

df = pd.DataFrame(filas_csv)

df = df.sort_values(
    ['ngrama', 'modelo', 'score_ranking'],
    ascending=[True, True, False]
).reset_index(drop=True)

df['rank'] = (
    df.groupby(['ngrama', 'modelo'])['score_ranking']
      .rank(ascending=False, method='first')
      .astype(int)
)

cols = ['rank', 'ngrama', 'modelo', 'reduccion',
        'score_ranking', 'silhouette', 'n_clusters',
        'n_ruido', 'pct_ruido', 'inercia', 'codo_k', 'hiperparametros']
df = df[cols]

path_ranking = os.path.join(DIR_BASE, 'ranking_completo.csv')
df.to_csv(path_ranking, index=False, encoding='utf-8-sig')
print(f'\nRanking completo guardado: {path_ranking}')

df_mejores = (
    df[df['rank'] == 1]
      .sort_values(['ngrama', 'modelo'])
      .reset_index(drop=True)
)
path_mejores = os.path.join(DIR_BASE, 'mejores_modelos.csv')
df_mejores.to_csv(path_mejores, index=False, encoding='utf-8-sig')
print(f'Mejores modelos guardados: {path_mejores}')

etiquetas_mejores = {}
for _, row in df_mejores.iterrows():
    key = f"{row['ngrama']}|{row['modelo']}|{row['reduccion']}|{row['hiperparametros']}"
    if key in etiquetas_dict:
        etiquetas_mejores[key] = etiquetas_dict[key]

path_etiq = os.path.join(DIR_BASE, 'etiquetas_mejores.json')
with open(path_etiq, 'w', encoding='utf-8') as f:
    json.dump(etiquetas_mejores, f)
print(f'Etiquetas guardadas: {path_etiq}')

print('\n--- RESUMEN: MEJORES MODELOS POR (NGRAMA, MODELO) ---')
print(df_mejores[['ngrama', 'modelo', 'reduccion', 'score_ranking',
                   'silhouette', 'n_clusters', 'n_ruido',
                   'hiperparametros']].to_string(index=False))
print('\nClustering completado.')