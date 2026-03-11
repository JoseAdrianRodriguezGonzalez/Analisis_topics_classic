'''
Modelado de topicos con Latent Dirichlet Allocation (LDA)
Evaluacion: Coherence Score y Perplexity
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

# --------------------------------------------
# Configuración general
# --------------------------------------------

sns.set_theme(style='whitegrid', palette='muted')
PALETTE = sns.color_palette('tab10')

DIR_DATOS    = '../data/processed/'
DIR_GRAFICAS = '../data/clusterizacion/'
PATH_NLP     = DIR_DATOS + 'data_nlp.csv'

os.makedirs(DIR_GRAFICAS, exist_ok=True)

N_TOPICOS_RANGO   = range(2, 11)   # valores de n_topics a evaluar
N_PALABRAS_TOP    = 10             # palabras más representativas por topico


# --------------------------------------------
# Funciones de carga
# --------------------------------------------

def cargar_corpus(path):
    '''
    Carga los comentarios limpios y elimina nulos/vacíos
    Devuelve la lista de textos y los metadatos alineados
    '''
    df = pd.read_csv(path)
    mask = df['comentario_cleaned'].notna() & (df['comentario_cleaned'].str.strip() != '')
    return df[mask]['comentario_cleaned'].tolist(), df[mask].reset_index(drop=True)


# --------------------------------------------
# Vectorizacion BoW para LDA
# LDA trabaja con conteos crudos, NO con TF-IDF
# --------------------------------------------

def vectorizar_bow(corpus, ngram_range=(1, 1), min_df=2):
    '''
    LDA requiere conteos enteros (BoW), no TF-IDF
    min_df=2: ignora palabras que aparecen en menos de 2 documentos
    ngram_range: (1,1) unigramas, (1,2) uni+bigramas
    '''
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(corpus)
    vocabulario = vectorizer.get_feature_names_out()
    return X, vocabulario, vectorizer


# --------------------------------------------
# Funciones de evaluacion
# --------------------------------------------

def calcular_perplexity(corpus_bow, n_rango):
    '''
    Perplexity: qué tan bien el modelo predice una muestra
    Menor perplexity = mejor modelo
    Se calcula directamente desde sklearn LDA
    '''
    perplexities = []
    for n in n_rango:
        modelo = LatentDirichletAllocation(
            n_components=n,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        modelo.fit(corpus_bow)
        perplexities.append(modelo.perplexity(corpus_bow))
        print(f'    n_topics={n} | perplexity={perplexities[-1]:.2f}')
    return perplexities


def calcular_coherence_umass(corpus_bow, vocabulario, n_rango, top_n=10):
    '''
    Coherence Score UMass (disponible sin gensim)
    Mide co-ocurrencia de las palabras top de cada topico
    Mayor coherence (menos negativo) = topicos más coherentes

    Para cada par (w_i, w_j) en el top del topico:
    C = log( (D(w_i, w_j) + 1) / D(w_j) )
    donde D(w) = documentos que contienen w
    '''
    # Matriz densa de co-ocurrencias (documentos x vocabulario)
    X_dense = corpus_bow.toarray().astype(bool).astype(int)
    n_docs = X_dense.shape[0]

    # Frecuencia de documento por término
    df_term = X_dense.sum(axis=0)  # vector (|V|,)

    # Co-ocurrencia: cuántos docs tienen ambas palabras i y j
    # X_dense.T @ X_dense da la matriz de co-ocurrencia (|V| x |V|)
    cooc = X_dense.T @ X_dense  # shape (|V|, |V|)

    coherences = []
    for n in n_rango:
        modelo = LatentDirichletAllocation(
            n_components=n,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        modelo.fit(corpus_bow)

        scores_topicos = []
        for topico in modelo.components_:
            # Índices de las top_n palabras del topico
            top_idx = np.argsort(topico)[::-1][:top_n]
            score = 0
            pares = 0
            for i in range(len(top_idx)):
                for j in range(i + 1, len(top_idx)):
                    wi = top_idx[i]
                    wj = top_idx[j]
                    d_wi_wj = cooc[wi, wj]
                    d_wj    = df_term[wj]
                    if d_wj > 0:
                        score += np.log((d_wi_wj + 1) / d_wj)
                        pares += 1
            scores_topicos.append(score / pares if pares > 0 else 0)

        coherences.append(np.mean(scores_topicos))
        print(f'    n_topics={n} | coherence UMass={coherences[-1]:.4f}')

    return coherences


# --------------------------------------------
# Funciones de graficacion - evaluacion
# --------------------------------------------

def graficar_perplexity(n_rango, perplexities, nombre, dir_out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(n_rango), perplexities, marker='o', linewidth=2, color=PALETTE[2])
    ax.set_title(f'Perplexity por numero de topicos - {nombre}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Numero de topicos')
    ax.set_ylabel('Perplexity (menor = mejor)')
    ax.set_xticks(list(n_rango))
    plt.tight_layout()
    plt.savefig(f'{dir_out}lda_perplexity_{nombre}.png', dpi=150)
    plt.close()


def graficar_coherence(n_rango, coherences, nombre, dir_out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(n_rango), coherences, marker='s', linewidth=2, color=PALETTE[3])
    ax.set_title(f'Coherence Score (UMass) - {nombre}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Numero de topicos')
    ax.set_ylabel('Coherence (mayor = mejor)')
    ax.set_xticks(list(n_rango))
    plt.tight_layout()
    plt.savefig(f'{dir_out}lda_coherence_{nombre}.png', dpi=150)
    plt.close()


def graficar_topicos(modelo, vocabulario, n_topicos, n_palabras, nombre, dir_out):
    '''
    Heatmap de peso de las top palabras por topico
    '''
    datos = []
    for t_idx, topico in enumerate(modelo.components_):
        top_idx = np.argsort(topico)[::-1][:n_palabras]
        for idx in top_idx:
            datos.append({
                'topico'  : f'Topico {t_idx}',
                'palabra' : vocabulario[idx],
                'peso'    : topico[idx]
            })

    df_plot = pd.DataFrame(datos)
    pivot   = df_plot.pivot(index='palabra', columns='topico', values='peso').fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, n_topicos * 1.5), max(6, n_palabras * 0.5)))
    sns.heatmap(pivot, annot=False, cmap='YlOrRd', linewidths=0.3, ax=ax)
    ax.set_title(f'Palabras por topico - {nombre} (n={n_topicos})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Topico')
    ax.set_ylabel('Palabra')
    plt.tight_layout()
    plt.savefig(f'{dir_out}lda_heatmap_{nombre}_n{n_topicos}.png', dpi=150)
    plt.close()


def graficar_distribucion_topicos(doc_topicos, meta, variable, nombre, n_topicos, dir_out):
    '''
    Distribucion demográfica por topico dominante
    '''
    df = meta.copy()
    df['topico_dominante'] = doc_topicos.argmax(axis=1)

    if variable == 'edad':
        bins   = [0, 25, 35, 50, 100]
        labels = ['18-25', '26-35', '36-50', '50+']
        df['edad_rango'] = pd.cut(df['edad'], bins=bins, labels=labels, right=False)
        col = 'edad_rango'
    else:
        col = variable

    tabla     = df.groupby(['topico_dominante', col]).size().unstack(fill_value=0)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    tabla_pct.plot(kind='bar', stacked=True, ax=ax,
                   colormap='tab10', edgecolor='white', linewidth=0.5)
    ax.set_title(f'Distribucion de {variable} por topico - {nombre}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Topico dominante')
    ax.set_ylabel('Porcentaje (%)')
    ax.legend(title=variable, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{dir_out}lda_demo_{variable}_{nombre}_n{n_topicos}.png', dpi=150)
    plt.close()


def exportar_topicos(modelo, vocabulario, n_palabras, nombre, n_topicos, dir_out):
    '''
    Exporta un CSV con las top palabras por topico
    '''
    filas = []
    for t_idx, topico in enumerate(modelo.components_):
        top_idx = np.argsort(topico)[::-1][:n_palabras]
        for rank, idx in enumerate(top_idx):
            filas.append({
                'topico'  : f'Topico_{t_idx}',
                'rank'    : rank + 1,
                'palabra' : vocabulario[idx],
                'peso'    : topico[idx]
            })
    pd.DataFrame(filas).to_csv(
        f'{dir_out}lda_topicos_{nombre}_n{n_topicos}.csv',
        index=False, encoding='utf-8-sig'
    )


def exportar_asignaciones(doc_topicos, meta, nombre, n_topicos, dir_out):
    '''
    Exporta la distribucion de topicos por documento
    '''
    df = meta.copy()
    for t in range(n_topicos):
        df[f'prob_topico_{t}'] = doc_topicos[:, t]
    df['topico_dominante'] = doc_topicos.argmax(axis=1)
    df.to_csv(
        f'{dir_out}lda_asignaciones_{nombre}_n{n_topicos}.csv',
        index=False, encoding='utf-8-sig'
    )


# ======================================================
# CONFIGURACION PRINCIPAL
# ======================================================

CONFIGURACIONES = {
    'unigramas'    : (1, 1),
    'uni_bigramas' : (1, 2),
}

# Peso coherence vs perplexity en el score combinado
# coherence mayor = mejor, perplexity menor = mejor
# Se normalizan ambos a [0,1] antes de combinar
ALPHA_LDA = 0.6   # 60% coherence, 40% perplexity invertida

DIR_LDA = '../data/LDA/'
os.makedirs(DIR_LDA, exist_ok=True)

corpus_textos, meta_completo = cargar_corpus(PATH_NLP)
print(f'Documentos validos: {len(corpus_textos)}')

ranking_lda = []   # tabla resumen de todas las evaluaciones

for nombre, ngram_range in CONFIGURACIONES.items():

    print(f'\nLDA - {nombre}  ngram_range={ngram_range}')

    X_bow, vocabulario, vectorizer = vectorizar_bow(corpus_textos, ngram_range=ngram_range)
    print(f'  Vocabulario: {len(vocabulario)} terminos | Docs: {X_bow.shape[0]}')

    # Evaluacion exhaustiva
    print('  Calculando perplexity...')
    perplexities = calcular_perplexity(X_bow, N_TOPICOS_RANGO)

    print('  Calculando coherence UMass...')
    coherences = calcular_coherence_umass(X_bow, vocabulario, N_TOPICOS_RANGO)

    # Graficas de evaluacion
    graficar_perplexity(N_TOPICOS_RANGO, perplexities, nombre, DIR_LDA)
    graficar_coherence(N_TOPICOS_RANGO, coherences, nombre, DIR_LDA)

    # Score combinado para seleccionar n optimo automaticamente
    # Normalizar perplexity: invertir y escalar a [0,1] (menor perplexity = mejor)
    perp_arr = np.array(perplexities)
    perp_norm = 1 - (perp_arr - perp_arr.min()) / (perp_arr.max() - perp_arr.min() + 1e-9)

    # Normalizar coherence a [0,1] (mayor = mejor)
    coh_arr  = np.array(coherences)
    coh_norm = (coh_arr - coh_arr.min()) / (coh_arr.max() - coh_arr.min() + 1e-9)

    scores_combinados = ALPHA_LDA * coh_norm + (1 - ALPHA_LDA) * perp_norm

    ns = list(N_TOPICOS_RANGO)
    n_optimo = ns[int(np.argmax(scores_combinados))]

    print(f'  n optimo seleccionado automaticamente: {n_optimo}')

    # Guardar resultados de evaluacion en ranking
    for i, n in enumerate(ns):
        ranking_lda.append({
            'configuracion'   : nombre,
            'ngram_range'     : str(ngram_range),
            'n_topicos'       : n,
            'perplexity'      : round(perplexities[i], 4),
            'coherence_umass' : round(coherences[i], 6),
            'score_combinado' : round(scores_combinados[i], 6),
            'es_optimo'       : (n == n_optimo),
        })

    # Modelo final con n optimo
    print(f'  Entrenando modelo final (n={n_optimo})...')
    modelo_final = LatentDirichletAllocation(
        n_components=n_optimo,
        random_state=42,
        max_iter=50,
        learning_method='batch'
    )
    modelo_final.fit(X_bow)
    doc_topicos = modelo_final.transform(X_bow)

    # Graficas de resultados del modelo optimo
    graficar_topicos(modelo_final, vocabulario, n_optimo, N_PALABRAS_TOP, nombre, DIR_LDA)

    for variable in ['genero', 'lugar', 'edad']:
        graficar_distribucion_topicos(doc_topicos, meta_completo,
                                      variable, nombre, n_optimo, DIR_LDA)

    # Exportar topicos y asignaciones del modelo optimo
    exportar_topicos(modelo_final, vocabulario, N_PALABRAS_TOP, nombre, n_optimo, DIR_LDA)
    exportar_asignaciones(doc_topicos, meta_completo, nombre, n_optimo, DIR_LDA)
    print(f'  Resultados guardados en {DIR_LDA}')

# Exportar ranking completo de evaluacion
df_ranking = pd.DataFrame(ranking_lda)
df_ranking = df_ranking.sort_values(
    ['configuracion', 'score_combinado'], ascending=[True, False]
).reset_index(drop=True)
df_ranking.to_csv(os.path.join(DIR_LDA, 'ranking_lda.csv'), index=False, encoding='utf-8-sig')

print('\n--- RESUMEN: MEJORES N POR CONFIGURACION ---')
mejores = df_ranking[df_ranking['es_optimo']]
print(mejores[['configuracion', 'n_topicos', 'perplexity',
               'coherence_umass', 'score_combinado']].to_string(index=False))

print('\nModelado de topicos completado.')