'''
AnalisisCluster.py
------------------
Analisis interno de clusters seleccionados:
  - TF-IDF por cluster: palabras que caracterizan cada grupo (punto 7)
  - Documentos representativos: textos mas cercanos al centroide (punto 8)

Reutiliza funciones de Vectorization.py:
  calcular_BoW, calcular_tf, calcular_idf, calcular_tf_idf, normalizacion_l2

Lee:
  data/clusterizacion/mejores_modelos.csv
  data/clusterizacion/etiquetas_mejores.json
  data/processed/data_nlp.csv
  data/processed/rankings_*.csv      (vocabularios)
  data/processed/TF_IDF_normalizado_*.csv

Guarda en:
  data/analisis_clusters/
    tfidf_{ngrama}_{modelo}_c{n}.csv  - ranking TF-IDF local por cluster
    representativos_{ngrama}_{modelo}.csv - docs mas cercanos al centroide
    resumen_palabras_clave.csv        - tabla resumen consolidada
'''

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# Importar funciones de Vectorization.py (mismo directorio model/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Vectorization import (calcular_tf, calcular_idf, calcular_tf_idf,
                            normalizacion_l2, generar_ngrams, calcular_BoW)

# ======================================================
# CONFIGURACION
# ======================================================

DIR_DATOS = '../data/processed/'
DIR_BASE  = '../data/clusterizacion/'
DIR_OUT   = '../data/analisis_clusters/'
PATH_NLP  = DIR_DATOS + 'data_nlp.csv'

ARCHIVOS_VOCAB = {
    'unigramas': DIR_DATOS + 'rankings_unigrams.csv',
    'bigramas' : DIR_DATOS + 'rankings_bigrams.csv',
    'trigramas': DIR_DATOS + 'rankings_trigrams.csv',
}

ARCHIVOS_TFIDF = {
    'unigramas': DIR_DATOS + 'TF_IDF_normalizado_unigramas.csv',
    'bigramas' : DIR_DATOS + 'TF_IDF_normalizado_bigramas.csv',
    'trigramas': DIR_DATOS + 'TF_IDF_normalizado_trigramas.csv',
}

N_VALORES = {'unigramas': 1, 'bigramas': 2, 'trigramas': 3}

# Modelos seleccionados para analisis
# Formato: ngrama -> lista de tuplas (modelo, reduccion)
MODELOS_SELECCIONADOS = {
    'unigramas': [('jerarquico', 'UMAP')],
    'bigramas' : [('jerarquico', 'UMAP'), ('dbscan', 'UMAP')],
    'trigramas': [('jerarquico', 'PCA')],
}

N_PALABRAS_TOP    = 15
N_REPRESENTATIVOS = 3

os.makedirs(DIR_OUT, exist_ok=True)


# ======================================================
# CARGA
# ======================================================

def cargar_tfidf_global(path):
    df   = pd.read_csv(path)
    mask = df.values.sum(axis=1) != 0
    X    = df.values[mask].astype(float)
    cols = df.columns.tolist()
    return X, np.where(mask)[0], cols


def cargar_metadatos(path, indices_validos):
    df = pd.read_csv(path)
    return df.iloc[indices_validos].reset_index(drop=True)


def cargar_corpus_valido(path_nlp, indices_validos):
    df = pd.read_csv(path_nlp)
    return df.iloc[indices_validos]['comentario_cleaned'].tolist()


# ======================================================
# PUNTO 7: TF-IDF INTERNO POR CLUSTER
# ======================================================

def tfidf_por_cluster(corpus, etiquetas, vocabulario, n):
    cluster_ids = sorted(set(etiquetas))
    resultados  = {}

    for cid in cluster_ids:
        mask_c   = np.where(etiquetas == cid)[0]
        corpus_c = [corpus[i] for i in mask_c]
        n_docs_c = len(corpus_c)

        BoW_c, _, _ = calcular_BoW(corpus_c, vocabulario, n=n)

        if BoW_c.shape[0] == 0:
            continue

        TF_c    = calcular_tf(BoW_c)
        IDF_c   = calcular_idf(BoW_c)
        TFIDF_c = calcular_tf_idf(TF_c, IDF_c)
        TFIDF_n = normalizacion_l2(TFIDF_c)

        centroide = TFIDF_n.mean(axis=0)
        top_idx   = np.argsort(centroide)[::-1][:N_PALABRAS_TOP]
        df_local  = (BoW_c > 0).sum(axis=0)

        top_terms = [
            {
                'cluster'        : cid,
                'rank'           : r + 1,
                'palabra'        : vocabulario[i],
                'score_centroide': round(centroide[i], 6),
                'idf_local'      : round(IDF_c[i], 6),
                'df_en_cluster'  : int(df_local[i]),
                'n_docs_cluster' : n_docs_c,
            }
            for r, i in enumerate(top_idx)
            if centroide[i] > 0
        ]
        resultados[cid] = top_terms

    return resultados


# ======================================================
# PUNTO 8: DOCUMENTOS REPRESENTATIVOS (ambos espacios)
# ======================================================

def documentos_representativos(X_orig, X_red, etiquetas, meta,
                                n_top=N_REPRESENTATIVOS):
    cluster_ids = sorted(set(etiquetas))
    filas       = []

    for cid in cluster_ids:
        mask_c = np.where(etiquetas == cid)[0]

        X_orig_c = X_orig[mask_c]
        X_red_c  = X_red[mask_c]

        centroide_orig = X_orig_c.mean(axis=0, keepdims=True)
        centroide_red  = X_red_c.mean(axis=0, keepdims=True)

        sims_orig = cosine_similarity(X_orig_c, centroide_orig).flatten()
        sims_red  = cosine_similarity(X_red_c,  centroide_red).flatten()

        top_orig  = set(np.argsort(sims_orig)[::-1][:n_top].tolist())
        top_red   = set(np.argsort(sims_red)[::-1][:n_top].tolist())
        coinciden = top_orig & top_red

        todos = sorted(top_orig | top_red, key=lambda i: sims_red[i], reverse=True)

        for local_idx in todos:
            global_idx = mask_c[local_idx]
            fila = {
                'cluster'             : cid,
                'indice_global'       : int(global_idx),
                'sim_coseno_tfidf'    : round(sims_orig[local_idx], 6),
                'sim_coseno_reducido' : round(sims_red[local_idx],  6),
                'top_en_tfidf'        : local_idx in top_orig,
                'top_en_reducido'     : local_idx in top_red,
                'coincide_en_ambos'   : local_idx in coinciden,
            }
            for col in ['comentario_cleaned', 'comentario', 'genero', 'lugar', 'edad']:
                if col in meta.columns:
                    fila[col] = meta.iloc[global_idx][col]
            filas.append(fila)

    return pd.DataFrame(filas)


def construir_resumen(tfidf_clusters, ngrama, modelo):
    filas = []
    for cid, terminos in tfidf_clusters.items():
        palabras_top = ', '.join([t['palabra'] for t in terminos[:10]])
        filas.append({
            'ngrama'        : ngrama,
            'modelo'        : modelo,
            'cluster'       : cid,
            'n_docs'        : terminos[0]['n_docs_cluster'],
            'palabras_clave': palabras_top,
        })
    return pd.DataFrame(filas)


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def reducir(X, metodo):
    if metodo == 'PCA':
        return PCA(n_components=2, random_state=42).fit_transform(X)
    if metodo == 'UMAP':
        return UMAP(n_components=2, random_state=42).fit_transform(X)
    if metodo == 'TSNE':
        perp = min(30, X.shape[0] - 1)
        return TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)


df_mejores = pd.read_csv(os.path.join(DIR_BASE, 'mejores_modelos.csv'))
with open(os.path.join(DIR_BASE, 'etiquetas_mejores.json'), 'r', encoding='utf-8') as f:
    etiquetas_mejores = json.load(f)

resumenes = []

for ngrama, seleccionados in MODELOS_SELECCIONADOS.items():
    n = N_VALORES[ngrama]

    # Cargar vocabulario, corpus y matriz TF-IDF global (se reusan para todos
    # los modelos del mismo ngrama)
    vocab_df    = pd.read_csv(ARCHIVOS_VOCAB[ngrama])
    vocabulario = vocab_df['ngram'].tolist()

    X_global, indices_validos, _ = cargar_tfidf_global(ARCHIVOS_TFIDF[ngrama])
    meta   = cargar_metadatos(PATH_NLP, indices_validos)
    corpus = cargar_corpus_valido(PATH_NLP, indices_validos)

    for modelo, reduccion in seleccionados:

        # Buscar la fila correspondiente en mejores_modelos.csv
        fila_df = df_mejores[
            (df_mejores['ngrama']    == ngrama) &
            (df_mejores['modelo']    == modelo) &
            (df_mejores['reduccion'] == reduccion)
        ]

        if fila_df.empty:
            print(f'No encontrado en mejores_modelos: {ngrama}|{modelo}|{reduccion}')
            continue

        fila     = fila_df.iloc[0]
        hiperpar = fila['hiperparametros']
        k        = int(fila['n_clusters'])

        key      = f"{ngrama}|{modelo}|{reduccion}|{hiperpar}"
        etiq_lst = etiquetas_mejores.get(key)

        if etiq_lst is None:
            print(f'Sin etiquetas para {key}, saltando.')
            continue

        etiquetas = np.array(etiq_lst)

        print(f'\n{ngrama} | {modelo} | {reduccion} | k={k} | {hiperpar}')

        # Punto 7: TF-IDF local por cluster
        print(f'  Calculando TF-IDF por cluster...')
        tfidf_clusters = tfidf_por_cluster(corpus, etiquetas, vocabulario, n)

        for cid, terminos in tfidf_clusters.items():
            pd.DataFrame(terminos).to_csv(
                os.path.join(DIR_OUT, f'tfidf_{ngrama}_{modelo}_c{cid}.csv'),
                index=False, encoding='utf-8-sig'
            )

        # Punto 8: documentos representativos en ambos espacios
        print(f'  Recalculando reduccion {reduccion}...')
        X_red = reducir(X_global, reduccion)

        print(f'  Identificando documentos representativos...')
        df_repr = documentos_representativos(X_global, X_red, etiquetas, meta)
        df_repr.to_csv(
            os.path.join(DIR_OUT, f'representativos_{ngrama}_{modelo}.csv'),
            index=False, encoding='utf-8-sig'
        )

        df_res = construir_resumen(tfidf_clusters, ngrama, modelo)
        resumenes.append(df_res)

        print(f'  Guardado en {DIR_OUT}')
        print(f'\n  Palabras clave por cluster ({ngrama} | {modelo}):')
        for cid, terminos in tfidf_clusters.items():
            palabras = ', '.join([t['palabra'] for t in terminos[:8]])
            n_docs   = terminos[0]['n_docs_cluster']
            print(f'    Cluster {cid} ({n_docs} docs): {palabras}')

if resumenes:
    pd.concat(resumenes, ignore_index=True).to_csv(
        os.path.join(DIR_OUT, 'resumen_palabras_clave.csv'),
        index=False, encoding='utf-8-sig'
    )

print('\nAnalisis de clusters completado.')
print(f'Archivos en: {DIR_OUT}')