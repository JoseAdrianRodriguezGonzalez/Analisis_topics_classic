'''
GraficadoClusterizacion.py
--------------------------
Lee mejores_modelos.csv y etiquetas_mejores.json generados por Clustering.py
y produce todas las graficas para el mejor modelo de cada (ngrama, modelo).

Modelos SELECCIONADOS (con etiquetas cualitativas completas):
  unigramas  | jerarquico | UMAP
  bigramas   | jerarquico | UMAP
  bigramas   | dbscan     | UMAP
  trigramas  | jerarquico | PCA

El resto de mejores_modelos.csv se grafican con C0, C1...

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
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

DIR_DATOS = '../data/processed/'
DIR_BASE = '../data/clusterizacion/'
PATH_NLP = DIR_DATOS + 'data_nlp.csv'

ARCHIVOS_TFIDF = {
    'unigramas': DIR_DATOS + 'TF_IDF_normalizado_unigramas.csv',
    'bigramas': DIR_DATOS + 'TF_IDF_normalizado_bigramas.csv',
    'trigramas': DIR_DATOS + 'TF_IDF_normalizado_trigramas.csv',
}

ARCHIVOS_RANKING = {
    'unigramas': DIR_DATOS + 'rankings_unigrams.csv',
    'bigramas': DIR_DATOS + 'rankings_bigrams.csv',
    'trigramas': DIR_DATOS + 'rankings_trigrams.csv',
}

CARPETA_NGRAMA = {
    'unigramas': 'Unigramas',
    'bigramas': 'Bigramas',
    'trigramas': 'Trigramas',
}

# Modelos que reciben etiquetas cualitativas
MODELOS_SELECCIONADOS = {
    'unigramas': [('jerarquico', 'UMAP')],
    'bigramas': [('jerarquico', 'UMAP'), ('dbscan', 'UMAP')],
    'trigramas': [('jerarquico', 'PCA')],
}

# ======================================================
# ETIQUETAS CUALITATIVAS
# ======================================================

ETIQUETAS_CUALITATIVAS = {
    ('unigramas', 'jerarquico'): {
        0: 'Ciudad con potencial pero insegura',
        1: 'Descuido urbano e insatisfacción ciudadana',
        2: 'Inseguridad vivida desde adentro',
        3: 'Ciudad linda pero con mucho por mejorar',
        4: 'Lugar promedio sin oferta turística clara',
        5: 'Inseguridad y contaminación como barreras',
        6: 'Gastronomía sí, turismo no',
        7: 'Peligro nocturno y deterioro urbano',
        8: 'Sin interés turístico, con pequeñas virtudes',
        9: 'Potencial turístico abandonado',
    },
    ('bigramas', 'jerarquico'): {
        0: 'Evaluación general de la ciudad',
        1: 'Potencial turístico desaprovechado',
        2: 'Experiencia de residentes foráneos',
        3: 'Crítica estructural y abandono urbano',
    },
    ('bigramas', 'dbscan'): {
        0: 'Percepción general de inseguridad',
        1: 'Experiencia limitada del visitante',
        2: 'Adaptación de residentes foráneos',
        3: 'Potencial turístico desaprovechado',
        4: 'Críticas a gobierno y seguridad',
        5: 'Infraestructura descuidada y desigual',
        6: 'Reacción emocional negativa',
    },
    ('trigramas', 'jerarquico'): {
        0: 'Ciudad promedio sin atractivos',
        1: 'Vida cotidiana condicionada por inseguridad',
        2: 'Reconocimiento gastronómico con limitaciones',
        3: 'Vínculos personales y experiences locales',
        4: 'Contaminación, inseguridad y percepción urbana',
        5: 'Inseguridad y falta de oferta recreativa',
        6: 'Potencial cultural y natural desaprovechado',
        7: 'Ciudad insegura y poco recreativa',
        8: 'Potencial turístico abandonado',
        9: 'Potencial gastronómico y cultural',
    },
}

sns.set_theme(style='whitegrid', palette='tab10', font='DejaVu Sans')
PALETTE_TAB10 = sns.color_palette('tab10', 20)

# ======================================================
# CREAR CARPETAS
# ======================================================

for carpeta_ng in CARPETA_NGRAMA.values():
    for red in ['PCA', 'UMAP', 'TSNE']:
        os.makedirs(os.path.join(DIR_BASE, carpeta_ng, red), exist_ok=True)
        os.makedirs(os.path.join(DIR_BASE, 'DEMOS', carpeta_ng, red), exist_ok=True)


# ======================================================
# HELPERS
# ======================================================

def cargar_matriz(path):
    df = pd.read_csv(path)
    mask = df.values.sum(axis=1) != 0
    return df.values[mask].astype(float), np.where(mask)[0], df.columns.tolist()


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
    return os.path.join(DIR_BASE, CARPETA_NGRAMA[ngrama], reduccion)


def es_seleccionado(ngrama, modelo):
    return modelo in [m for m, _ in MODELOS_SELECCIONADOS.get(ngrama, [])]


def get_etiq_dict(ngrama, modelo):
    return ETIQUETAS_CUALITATIVAS.get((ngrama, modelo), None)


def label_corto(cid, etiq_dict):
    '''Etiqueta para leyenda: "C0: Título corto"'''
    if etiq_dict and cid in etiq_dict:
        titulo = etiq_dict[cid]
        # Acortar a ~30 chars para que la leyenda no sea enorme
        if len(titulo) > 30:
            titulo = titulo[:28] + '…'
        return f'C{cid}: {titulo}'
    return f'C{cid}'


def label_largo(cid, etiq_dict):
    '''Etiqueta completa para eje Y'''
    if etiq_dict and cid in etiq_dict:
        return etiq_dict[cid]
    return f'C{cid}'


def clean_for_text(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = text.replace('"', '').replace("'", '').replace('“', '').replace('”', '').replace('—', '-')
    return text.strip()


# ======================================================
# GRAFICAS — scatter, silhouette, elbow, dendrograma
# ======================================================

def graficar_scatter_clusters(X_red, etiquetas, titulo, path_out,
                              reduccion, es_dbscan=False, etiq_dict=None):
    ids_validos = sorted([c for c in set(etiquetas) if c != -1])
    palette = sns.color_palette('tab10', n_colors=max(len(ids_validos), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    if es_dbscan:
        mask_r = etiquetas == -1
        if mask_r.any():
            ax.scatter(X_red[mask_r, 0], X_red[mask_r, 1],
                       c='#cccccc', s=45, alpha=0.5,
                       edgecolors='#999999', linewidths=0.4,
                       label='Ruido', zorder=2)

    for ci, cid in enumerate(ids_validos):
        mask = etiquetas == cid
        ax.scatter(X_red[mask, 0], X_red[mask, 1],
                   color=palette[ci % len(palette)], s=75, alpha=0.85,
                   edgecolors='white', linewidths=0.4,
                   label=label_corto(cid, etiq_dict), zorder=3)

    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f'{reduccion} — dimensión 1', fontsize=10)
    ax.set_ylabel(f'{reduccion} — dimensión 2', fontsize=10)
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, title_fontsize=9, framealpha=0.9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches='tight')
    plt.close()


def graficar_silhouette_detalle(X_red, etiquetas, titulo, path_out, etiq_dict=None):
    valores = silhouette_samples(X_red, etiquetas)
    promedio = valores.mean()
    ids = sorted(set(etiquetas))
    palette = sns.color_palette('tab10', n_colors=len(ids))

    fig, ax = plt.subplots(figsize=(9, 5))
    y = 0
    ticks_y, tick_labs = [], []

    for ci, cid in enumerate(ids):
        vals = np.sort(valores[etiquetas == cid])
        h = len(vals)
        ax.barh(range(y, y + h), vals, height=1.0,
                color=palette[ci % len(palette)],
                edgecolor='none', alpha=0.88)
        ticks_y.append(y + h // 2)
        tick_labs.append(label_largo(cid, etiq_dict))
        y += h + 4

    ax.axvline(promedio, color='crimson', linestyle='--', lw=1.8,
               label=f'Promedio: {promedio:.3f}')
    ax.set_yticks(ticks_y)
    ax.set_yticklabels(tick_labs, fontsize=7)
    ax.set_xlabel('Silhouette por muestra', fontsize=10)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches='tight')
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
    sns.lineplot(x=ks, y=iv, marker='o', linewidth=2,
                 color=PALETTE_TAB10[0], ax=ax)
    ax.axvline(codo_k, color='crimson', linestyle='--', lw=1.8,
               label=f'Codo k={codo_k}')
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel('Número de clusters (k)', fontsize=10)
    ax.set_ylabel('Inercia', fontsize=10)
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches='tight')
    plt.close()


# Dendrograma Original (Preservado)
def graficar_dendrograma(X, metodo, titulo, path_out, max_hojas=50):
    Z = linkage(X, method=metodo)
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=max_hojas,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel('Documentos', fontsize=10)
    ax.set_ylabel('Distancia', fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches='tight')
    plt.close()


# NUEVO V2: Dendrograma con Tópicos calculados dinámicamente
def graficar_dendrograma_etiquetado(X, etiquetas, etiq_dict, metodo, titulo, path_out, max_hojas=50):
    Z = linkage(X, method=metodo)
    fig, ax = plt.subplots(figsize=(16, 8))

    # 1. Dibujar el dendrograma normal
    R = dendrogram(Z, ax=ax, truncate_mode='lastp', p=max_hojas,
                   leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.7 * max(Z[:, 2]))

    # 2. Rastrear qué cluster domina en cada nodo del árbol (Mapeo de Scipy)
    n_samples = len(etiquetas)
    # Diccionario para guardar qué documentos hay en cada nodo (originales + fusionados)
    node_clusters = {i: [etiquetas[i]] for i in range(n_samples)}

    for i, row in enumerate(Z):
        izq, der = int(row[0]), int(row[1])
        # El nuevo nodo creado tiene el ID: n_samples + i
        node_clusters[n_samples + i] = node_clusters[izq] + node_clusters[der]

    # 3. Generar las nuevas etiquetas para las hojas visibles
    nuevas_etiquetas_eje = []

    for leaf_node in R['leaves']:
        docs_en_hoja = node_clusters[leaf_node]
        validos = [c for c in docs_en_hoja if c != -1]

        if not validos:
            validos = [-1]

        # Cluster más común en esta rama
        dom_c = Counter(validos).most_common(1)[0][0]

        if dom_c == -1:
            texto_tema = "Ruido"
        else:
            texto_tema = etiq_dict.get(dom_c, f"Tema {dom_c}")

        if len(texto_tema) > 35:
            texto_tema = texto_tema[:32] + "..."

        etiqueta_final = f"C{dom_c}: {texto_tema}\n({len(docs_en_hoja)} docs)"
        nuevas_etiquetas_eje.append(etiqueta_final)

    # 4. Inyectar nuestros textos en el eje X
    ax.set_xticklabels(nuevas_etiquetas_eje, rotation=45, ha='right', fontsize=9, fontweight='bold')

    ax.set_title(titulo, fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Distancia (Disimilitud)', fontsize=11)
    ax.set_xlabel('Tema predominante por rama', fontsize=12, labelpad=10)

    sns.despine(bottom=True)
    plt.tight_layout()
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================
# DEMOGRAFIA
# ======================================================

def graficar_demografia(meta, etiquetas, variable, titulo, path_out,
                        etiq_dict=None):
    df = meta.copy()
    df['cluster'] = etiquetas
    df = df[df['cluster'] != -1].copy()
    if df.empty:
        return

    if variable == 'edad':
        df['_col'] = pd.to_numeric(df['edad'], errors='coerce')
        df = df.dropna(subset=['_col'])
        df['_col'] = df['_col'].astype(int).astype(str)
        col_var = '_col'
        titulo_grafica = titulo.replace('edad', 'Edad')
        label_eje = 'Edad'
    elif variable == 'genero':
        df['_col'] = df['genero']
        col_var = '_col'
        titulo_grafica = titulo.replace('genero', 'Género')
        label_eje = 'Género'
    else:
        df['_col'] = df['lugar']
        col_var = '_col'
        titulo_grafica = titulo.replace('lugar', 'Ciudad de Origen')
        label_eje = 'Ciudad de Origen'

    if col_var not in df.columns or df[col_var].isna().all():
        return

    tabla = df.groupby([col_var, 'cluster']).size().unstack(fill_value=0)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    ids_cluster = sorted(tabla_pct.columns.tolist())
    n_clusters = len(ids_cluster)
    palette = sns.color_palette('tab10', n_colors=n_clusters)
    color_map = {cid: palette[i] for i, cid in enumerate(ids_cluster)}

    n_grupos = len(tabla_pct)
    fig, ax = plt.subplots(figsize=(12, max(4, n_grupos * 1.1 + 1.5)))

    lefts = np.zeros(n_grupos)
    grupos = tabla_pct.index.tolist()

    for cid in ids_cluster:
        vals = tabla_pct[cid].values if cid in tabla_pct.columns else np.zeros(n_grupos)
        ax.barh(grupos, vals, left=lefts,
                color=color_map[cid],
                edgecolor='white', linewidth=0.6,
                label=label_corto(cid, etiq_dict))
        lefts += vals

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.set_xlabel('Porcentaje de participación', fontsize=11)
    ax.set_ylabel(label_eje, fontsize=11)
    ax.set_title(titulo_grafica, fontsize=14, fontweight='bold', pad=14)

    ax.legend(title='Temas Identificados',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8.5, title_fontsize=10,
              framealpha=0.95, edgecolor='#cccccc')

    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# FRECUENCIAS DE NGRAM POR CLUSTER
# ======================================================

def graficar_ngram_por_cluster(X, vocabulario, etiquetas, ngrama, modelo,
                               reduccion, dir_red, es_dbscan=False,
                               etiq_dict=None, top_n=15):
    ids_cluster = sorted([c for c in set(etiquetas) if c != -1])
    cmap = plt.get_cmap('plasma')

    for cid in ids_cluster:
        mask = etiquetas == cid
        if mask.sum() == 0:
            continue

        pesos = X[mask].sum(axis=0)
        indices_top = np.argsort(pesos)[-top_n:][::-1]
        palabras = [vocabulario[i] for i in indices_top]
        valores = [pesos[i] for i in indices_top]

        orden = np.argsort(valores)
        palabras = [palabras[i] for i in orden]
        valores = [valores[i] for i in orden]

        n = len(palabras)
        colores = [cmap(0.2 + 0.6 * (i / max(n - 1, 1))) for i in range(n)]

        titulo_cluster = label_largo(cid, etiq_dict)
        fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45 + 1.5)))
        bars = ax.barh(palabras, valores, color=colores,
                       edgecolor='white', linewidth=0.5)

        ax.set_title(f'Top {top_n} {ngrama} — C{cid}: {titulo_cluster}\n'
                     f'({modelo.upper()}, {reduccion})',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Peso TF-IDF acumulado', fontsize=10)
        ax.set_ylabel(ngrama.capitalize(), fontsize=10)

        for bar, val in zip(bars, valores):
            ax.text(bar.get_width() + max(valores) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha='left', fontsize=8)

        ax.set_xlim(0, max(valores) * 1.18)
        ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='gray')
        ax.set_axisbelow(True)
        sns.despine()
        plt.tight_layout()

        fname = f'ngram_cluster{cid}_{ngrama}_{modelo}.png'
        plt.savefig(os.path.join(dir_red, fname), dpi=150, bbox_inches='tight')
        plt.close()


# ======================================================
# NUEVO: BOLSA DE PALABRAS COMO HEATMAP
# ======================================================

def graficar_heatmap_palabras_cluster(X, vocabulario, etiquetas, etiq_dict, titulo, path_out, top_n_words=10):
    ids_validos = sorted([c for c in set(etiquetas) if c != -1])
    if not ids_validos or len(ids_validos) < 2:
        return

    cluster_weights = []
    cluster_keywords_list = []

    for cid in ids_validos:
        mask = etiquetas == cid
        if mask.sum() == 0:
            continue

        pesos = X[mask].sum(axis=0)
        indices_top = np.argsort(pesos)[-top_n_words:][::-1]

        cluster_keywords_list.extend([vocabulario[i] for i in indices_top])
        cluster_weights.append(pesos)

    matriz_completa = np.stack(cluster_weights)

    df_pesos = pd.DataFrame(matriz_completa, columns=vocabulario).T
    df_pesos.columns = [label_corto(c, etiq_dict) for c in ids_validos]

    palabras_unicas = list(set(cluster_keywords_list))
    if not palabras_unicas:
        return

    df_subset = df_pesos.loc[palabras_unicas]

    df_subset['cluster_max'] = df_subset.idxmax(axis=1)
    df_subset = df_subset.sort_values(by=['cluster_max'])
    df_final = df_subset.drop(columns=['cluster_max'])

    n_palabras_final = len(df_final)
    fig, ax = plt.subplots(figsize=(12, max(8, n_palabras_final * 0.3 + 2)))

    df_final_norm = df_final.div(df_final.sum(axis=0), axis=1) * 100

    sns.heatmap(df_final_norm, cmap='plasma', ax=ax, annot=False,
                fmt='.2f', cbar_kws={'label': 'Peso Relativo Global (%)'})

    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xlabel('Temas Identificados (Modelo Seleccionado)', fontsize=12)
    ax.set_ylabel('Palabras Clave (Top N Global)', fontsize=12)

    plt.tight_layout()
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

df_mejores = pd.read_csv(os.path.join(DIR_BASE, 'mejores_modelos.csv'))

with open(os.path.join(DIR_BASE, 'etiquetas_mejores.json'), 'r', encoding='utf-8') as f:
    etiquetas_mejores = json.load(f)

for _, fila in df_mejores.iterrows():
    ngrama = fila['ngrama']
    modelo = fila['modelo']
    reduccion = fila['reduccion']
    hiperpar = fila['hiperparametros']
    n_clusters = int(fila['n_clusters'])
    codo_k = fila.get('codo_k', None)

    key = f"{ngrama}|{modelo}|{reduccion}|{hiperpar}"
    etiq_lst = etiquetas_mejores.get(key)
    if etiq_lst is None:
        print(f'Sin etiquetas para {key}, saltando.')
        continue

    etiquetas = np.array(etiq_lst)
    es_dbscan = (modelo == 'dbscan')
    seleccionado = es_seleccionado(ngrama, modelo)
    etiq_dict = get_etiq_dict(ngrama, modelo) if seleccionado else None

    print(f'{ngrama} | {modelo} | {reduccion} | {hiperpar} | seleccionado={seleccionado}')

    X, idx_val, vocab = cargar_matriz(ARCHIVOS_TFIDF[ngrama])
    meta = cargar_metadatos(PATH_NLP, idx_val)
    X_red = reducir(X, reduccion)

    dir_red = dir_grafica(ngrama, reduccion)
    prefijo = f'{ngrama}_{modelo}'

    # -- Scatter (Preservado) --
    graficar_scatter_clusters(
        X_red, etiquetas,
        titulo=f'{modelo.upper()} ({hiperpar}) — {ngrama} [{reduccion}]',
        path_out=os.path.join(dir_red, f'scatter_{prefijo}.png'),
        reduccion=reduccion, es_dbscan=es_dbscan, etiq_dict=etiq_dict,
    )

    # -- Silhouette (Preservado) --
    etiq_v = etiquetas[etiquetas != -1] if es_dbscan else etiquetas
    X_v = X_red[etiquetas != -1] if es_dbscan else X_red
    if len(set(etiq_v)) >= 2:
        graficar_silhouette_detalle(
            X_v, etiq_v,
            titulo=f'Silhouette — {modelo.upper()} {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_red, f'silhouette_{prefijo}.png'),
            etiq_dict=etiq_dict,
        )

    # -- Elbow (solo kmeans, Preservado) --
    if modelo == 'kmeans' and not pd.isna(codo_k):
        graficar_elbow(
            X_red, int(codo_k),
            titulo=f'Elbow — K-means {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_red, f'elbow_{prefijo}.png'),
        )

    # -- Dendrograma (solo jerarquico, Preservado original) --
    if modelo == 'jerarquico':
        metodo_jer = hiperpar.split('metodo=')[-1]
        graficar_dendrograma(
            X_red, metodo_jer,
            titulo=f'Dendrograma (Original) — Jerárquico {metodo_jer} {ngrama} [{reduccion}]',
            path_out=os.path.join(dir_red, f'dendrograma_{prefijo}_orig.png'),
        )

    # -- Dendrograma V2 (Etiquetado, NUEVO) --
    if modelo == 'jerarquico' and seleccionado:
        metodo_jer = hiperpar.split('metodo=')[-1]
        graficar_dendrograma_etiquetado(
            X_red, etiquetas, etiq_dict, metodo_jer,
            titulo=f'Dendrograma de Tópicos (V2) — {modelo.capitalize()} {metodo_jer} {ngrama}',
            path_out=os.path.join(dir_red, f'dendrograma_{prefijo}_etiquetado.png'),
            max_hojas=30
        )

    # -- Heatmap de Palabras Clave (NUEVO) --
    if seleccionado:
        titulo_heatmap = (
            f'Matriz de Calor (Heatmap) — Palabras Clave por Tema\n'
            f'({modelo.upper()}, {reduccion}) — {ngrama.capitalize()}'
        )
        graficar_heatmap_palabras_cluster(
            X, vocab, etiquetas, etiq_dict,
            titulo=titulo_heatmap,
            path_out=os.path.join(dir_red, f'heatmap_palabras_{prefijo}.png'),
            top_n_words=10
        )

    # -- Demografia (Preservado) --
    dir_demos = os.path.join(DIR_BASE, 'DEMOS', CARPETA_NGRAMA[ngrama], reduccion)
    for var in ['genero', 'lugar', 'edad']:
        titulo_demo = (
            f'Distribución de Temas por {var.capitalize()} — '
            f'{modelo.upper()} {ngrama} [{reduccion}]'
        )
        graficar_demografia(
            meta, etiquetas, var,
            titulo=titulo_demo,
            path_out=os.path.join(dir_demos, f'demo_{var}_{prefijo}.png'),
            etiq_dict=etiq_dict,
        )

    # -- Frecuencias de ngram por cluster (solo seleccionados, Preservado) --
    if seleccionado:
        graficar_ngram_por_cluster(
            X, vocab, etiquetas, ngrama, modelo, reduccion,
            dir_red=dir_red, es_dbscan=es_dbscan,
            etiq_dict=etiq_dict, top_n=15,
        )

print('\nGraficación completada.')