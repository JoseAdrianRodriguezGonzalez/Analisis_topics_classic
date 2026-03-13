'''
GraficadoClusterizacion.py
--------------------------
Lee mejores_modelos.csv y etiquetas_mejores.json generados por Clustering.py
y produce todas las graficas para el mejor modelo de cada (ngrama, modelo).

Estructura de salida:
  data/clusterizacion/
    Unigramas/  PCA/  UMAP/  TSNE/
    Bigramas/   PCA/  UMAP/  TSNE/
    Trigramas/  PCA/  UMAP/  TSNE/
    DEMOS/
'''

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans as _KM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
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

# Mapa de nombre de ngrama a nombre de carpeta (respeta el renombre que hiciste)
CARPETA_NGRAMA = {
    'unigramas': 'Unigramas',
    'bigramas' : 'Bigramas',
    'trigramas': 'Trigramas',
}

sns.set_theme(style='whitegrid', palette='muted')
PALETTE = sns.color_palette('tab10')


# ======================================================
# CREAR CARPETAS
# ======================================================

dir_demos = os.path.join(DIR_BASE, 'DEMOS')
os.makedirs(dir_demos, exist_ok=True)

for carpeta_ng in CARPETA_NGRAMA.values():
    for red in ['PCA', 'UMAP', 'TSNE']:
        os.makedirs(os.path.join(DIR_BASE, carpeta_ng, red), exist_ok=True)


# ======================================================
# CARGA DE DATOS
# ======================================================

def cargar_matriz(path):
    df   = pd.read_csv(path)
    mask = df.values.sum(axis=1) != 0
    return df.values[mask].astype(float), np.where(mask)[0]


def cargar_metadatos(path, indices_validos):
    return pd.read_csv(path).iloc[indices_validos].reset_index(drop=True)


def reducir(X, metodo):
    if metodo == 'PCA':
        return PCA(n_components=2, random_state=42).fit_transform(X)
    if metodo == 'UMAP':
        return UMAP(n_components=2, random_state=42).fit_transform(X)
    if metodo == 'TSNE':
        perp = min(30, X.shape[0] - 1)
        return TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)


def dir_grafica(ngrama, reduccion):
    '''Devuelve la carpeta correcta: clusterizacion/Unigramas/PCA/ etc.'''
    return os.path.join(DIR_BASE, CARPETA_NGRAMA[ngrama], reduccion)


# ======================================================
# GRAFICAS
# ======================================================

def graficar_scatter_clusters(X_red, etiquetas, titulo, path_out, reduccion, es_dbscan=False):
    df_plot = pd.DataFrame({'x': X_red[:, 0], 'y': X_red[:, 1],
                            'cluster': etiquetas})
    fig, ax = plt.subplots(figsize=(8, 6))

    if es_dbscan:
        ruido = df_plot[df_plot['cluster'] == -1]
        if len(ruido):
            ax.scatter(ruido['x'], ruido['y'], c='lightgray', s=55,
                       alpha=0.6, edgecolors='gray', lw=0.4,
                       label='Ruido (-1)', zorder=2)
        df_plot = df_plot[df_plot['cluster'] != -1].copy()

    df_plot['cluster'] = df_plot['cluster'].astype(str)
    sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster',
                    palette='tab10', s=80, alpha=0.88, ax=ax, zorder=3)

    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel(f'{reduccion} dim 1')
    ax.set_ylabel(f'{reduccion} dim 2')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close()


def graficar_silhouette_detalle(X_red, etiquetas, titulo, path_out):
    valores  = silhouette_samples(X_red, etiquetas)
    promedio = valores.mean()
    ids      = sorted(set(etiquetas))

    fig, ax = plt.subplots(figsize=(8, 5))
    y = 0
    ticks_y = []
    for ci, cid in enumerate(ids):
        vals = np.sort(valores[etiquetas == cid])
        h    = len(vals)
        ax.barh(range(y, y + h), vals, height=1.0,
                color=PALETTE[ci % len(PALETTE)], edgecolor='none', alpha=0.85)
        ticks_y.append(y + h // 2)
        y += h + 5

    ax.axvline(promedio, color='red', linestyle='--', lw=1.5,
               label=f'Promedio: {promedio:.3f}')
    ax.set_yticks(ticks_y)
    ax.set_yticklabels([f'C{cid}' for cid in ids])
    ax.set_xlabel('Silhouette por muestra')
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close()


def graficar_elbow(X_red, codo_k, titulo, path_out):
    inercias = {}
    for k in range(2, 11):
        m = _KM(n_clusters=k, random_state=42, n_init='auto')
        m.fit(X_red)
        inercias[k] = m.inertia_

    ks = sorted(inercias.keys())
    iv = [inercias[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, iv, marker='o', lw=2, color=PALETTE[0])
    ax.axvline(codo_k, color='red', linestyle='--', lw=1.5,
               label=f'Codo k={codo_k}')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Inercia')
    ax.set_xticks(ks)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close()


def graficar_dendrograma(X, metodo, titulo, path_out, max_hojas=50):
    Z = linkage(X, method=metodo)
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=max_hojas,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel('Documentos')
    ax.set_ylabel('Distancia')
    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close()


def graficar_demografia(meta, etiquetas, variable, titulo, path_out):
    df = meta.copy()
    df['cluster'] = etiquetas

    if variable == 'edad':
        bins   = [0, 25, 35, 50, 120]
        labels = ['18-25', '26-35', '36-50', '50+']
        df['edad_rango'] = pd.cut(df['edad'], bins=bins, labels=labels, right=False)
        col = 'edad_rango'
    else:
        col = variable

    df = df[df['cluster'] != -1]
    if df.empty:
        return

    tabla     = df.groupby(['cluster', col]).size().unstack(fill_value=0)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    tabla_pct.plot(kind='bar', stacked=True, ax=ax,
                   colormap='tab10', edgecolor='white', lw=0.5)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('%')
    ax.legend(title=variable, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close()


# ======================================================
# PIPELINE DE GRAFICACION
# ======================================================

df_mejores = pd.read_csv(os.path.join(DIR_BASE, 'mejores_modelos.csv'))

with open(os.path.join(DIR_BASE, 'etiquetas_mejores.json'), 'r', encoding='utf-8') as f:
    etiquetas_mejores = json.load(f)

for _, fila in df_mejores.iterrows():
    ngrama     = fila['ngrama']
    modelo     = fila['modelo']
    reduccion  = fila['reduccion']
    hiperpar   = fila['hiperparametros']
    n_clusters = int(fila['n_clusters'])
    codo_k     = fila.get('codo_k', None)

    key      = f"{ngrama}|{modelo}|{reduccion}|{hiperpar}"
    etiq_lst = etiquetas_mejores.get(key)
    if etiq_lst is None:
        print(f'Sin etiquetas para {key}, saltando.')
        continue

    etiquetas = np.array(etiq_lst)
    es_dbscan = (modelo == 'dbscan')

    print(f'{ngrama} | {modelo} | {reduccion} | {hiperpar}')

    X, idx_val = cargar_matriz(ARCHIVOS_TFIDF[ngrama])
    meta       = cargar_metadatos(PATH_NLP, idx_val)
    X_red      = reducir(X, reduccion)

    # Graficas de proyeccion: Unigramas/PCA/, Bigramas/UMAP/, etc.
    dir_red = dir_grafica(ngrama, reduccion)
    prefijo = f'{ngrama}_{modelo}'

    # Scatter
    graficar_scatter_clusters(
        X_red, etiquetas,
        titulo=f'{modelo.upper()} ({hiperpar}) - {ngrama} [{reduccion}]',
        path_out=os.path.join(dir_red, f'scatter_{prefijo}.png'),
        reduccion=reduccion,
        es_dbscan=es_dbscan,
    )

    # Silhouette detalle
    etiq_validas = etiquetas[etiquetas != -1] if es_dbscan else etiquetas
    X_validas    = X_red[etiquetas != -1]     if es_dbscan else X_red
    if len(set(etiq_validas)) >= 2:
        graficar_silhouette_detalle(
            X_validas, etiq_validas,
            titulo=f'Silhouette - {modelo.upper()} {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_red, f'silhouette_{prefijo}.png'),
        )

    # Elbow (solo K-means)
    if modelo == 'kmeans' and not pd.isna(codo_k):
        graficar_elbow(
            X_red, int(codo_k),
            titulo=f'Elbow - K-means {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_red, f'elbow_{prefijo}.png'),
        )

    # Dendrograma (solo jerarquico)
    if modelo == 'jerarquico':
        metodo_jer = hiperpar.split('metodo=')[-1]
        graficar_dendrograma(
            X_red, metodo_jer,
            titulo=f'Dendrograma - Jerarquico {metodo_jer} {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_red, f'dendrograma_{prefijo}.png'),
        )

    # Demografia en DEMOS/
    for var in ['genero', 'lugar', 'edad']:
        graficar_demografia(
            meta, etiquetas, var,
            titulo=f'{var.capitalize()} por cluster - {modelo.upper()} {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_demos, f'demo_{var}_{prefijo}.png'),
        )

print('\nGraficacion completada.')