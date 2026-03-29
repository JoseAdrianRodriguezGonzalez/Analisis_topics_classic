'''
GraficadoLDA.py
---------------
Genera todas las graficas de los modelos LDA optimos con seaborn.
Lee los CSVs exportados por TopicModeling.py.

Estructura de salida:
  data/clusterizacion/LDA/
    unigramas/
      lda_heatmap_unigramas.png
      lda_perplexity_unigramas.png
      lda_coherence_unigramas.png
      lda_distribucion_topicos_unigramas.png
      lda_probabilidades_unigramas.png
      demo_genero_unigramas.png
      demo_lugar_unigramas.png
      demo_edad_unigramas.png
      ngram_topico{n}_unigramas.png
    uni_bigramas/
      (mismas graficas)
'''

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# ======================================================
# CONFIGURACION
# ======================================================

DIR_LDA   = '../data/LDA/'
DIR_DATOS = '../data/processed/'
DIR_OUT   = '../data/clusterizacion/LDA/'
PATH_NLP  = DIR_DATOS + 'data_nlp.csv'

N_OPTIMO = {
    'unigramas'   : 10,
    'uni_bigramas': 7,
}

CARPETA = {
    'unigramas'   : 'unigramas',
    'uni_bigramas': 'uni_bigramas',
}

# ======================================================
# ETIQUETAS CUALITATIVAS
# ======================================================

ETIQUETAS_LDA = {
    'unigramas': {
        0: 'Ciudad peligrosa con uso funcional',
        1: 'Vivir aquí: inseguridad y adaptación',
        2: 'Salamanca: ciudad con aspectos negativos',
        3: 'Ciudad pequeña, oferta turística mínima',
        4: 'Transporte caro, ciudad insegura',
        5: 'Refinería, violencia e imagen urbana',
        6: 'Lugar accesible pero culturalmente limitado',
        7: 'Potencial percibido, gobierno cuestionado',
        8: 'Opiniones dispersas: descuido y contaminación',
        9: 'Ciudad con potencial, poco visitada',
    },
    'uni_bigramas': {
        0: 'Lugar contaminado, poco recomendable',
        1: 'Descuido urbano e infraestructura deteriorada',
        2: 'Ciudad con oferta puntual, imagen deteriorada',
        3: 'Potencial reconocido, visitada por vínculos',
        4: 'Inseguridad y mala imagen: rechazo explícito',
        5: 'Inseguridad vivida: salir es riesgo',
        6: 'Transporte deficiente, uso funcional',
    },
}

sns.set_theme(style='whitegrid', palette='tab10', font='DejaVu Sans')
PALETTE_TAB10 = sns.color_palette('tab10', 20)


# ======================================================
# CREAR CARPETAS
# ======================================================

for carpeta in CARPETA.values():
    os.makedirs(os.path.join(DIR_OUT, carpeta), exist_ok=True)
    os.makedirs(os.path.join(DIR_OUT, carpeta, 'demos'), exist_ok=True)


# ======================================================
# HELPERS
# ======================================================

def get_label(tid, nombre):
    return ETIQUETAS_LDA.get(nombre, {}).get(tid, f'Tópico {tid}')


def label_corto_lda(tid, nombre):
    '''T{n}: Título corto para leyenda'''
    full = get_label(tid, nombre)
    if len(full) > 30:
        full = full[:28] + '…'
    return f'T{tid}: {full}'


def path_out(nombre, archivo):
    return os.path.join(DIR_OUT, CARPETA[nombre], archivo)


# ======================================================
# EVALUACION: perplexity y coherence
# ======================================================

def graficar_perplexity(ranking, nombre, dir_salida):
    df = ranking[ranking['configuracion'] == nombre].sort_values('n_topicos')
    optimo = df[df['es_optimo']]['n_topicos'].values[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x='n_topicos', y='perplexity',
                 marker='o', linewidth=2, color=PALETTE_TAB10[2], ax=ax)
    ax.axvline(optimo, color='crimson', linestyle='--', lw=1.8,
               label=f'Óptimo n={optimo}')
    ax.set_title(f'Perplexity por número de tópicos — {nombre}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Número de tópicos')
    ax.set_ylabel('Perplexity (menor = mejor)')
    ax.set_xticks(df['n_topicos'].tolist())
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f'lda_perplexity_{nombre}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def graficar_coherence(ranking, nombre, dir_salida):
    df = ranking[ranking['configuracion'] == nombre].sort_values('n_topicos')
    optimo = df[df['es_optimo']]['n_topicos'].values[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x='n_topicos', y='coherence_umass',
                 marker='s', linewidth=2, color=PALETTE_TAB10[3], ax=ax)
    ax.axvline(optimo, color='crimson', linestyle='--', lw=1.8,
               label=f'Óptimo n={optimo}')
    ax.set_title(f'Coherence Score UMass — {nombre}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Número de tópicos')
    ax.set_ylabel('Coherence (mayor = mejor)')
    ax.set_xticks(df['n_topicos'].tolist())
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f'lda_coherence_{nombre}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# HEATMAP palabras x topico
# ======================================================

def graficar_heatmap(topicos_df, nombre, n, dir_salida):
    pivot = topicos_df.pivot(index='palabra', columns='topico', values='peso').fillna(0)

    col_labels = {}
    for col in pivot.columns:
        tid = int(str(col).replace('Topico_', ''))
        col_labels[col] = get_label(tid, nombre)[:25]
    pivot = pivot.rename(columns=col_labels)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.8), max(7, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=False, cmap='YlOrRd', linewidths=0.3,
                linecolor='#f0f0f0', ax=ax, cbar_kws={'shrink': 0.6})
    ax.set_title(f'Palabras clave por tópico — {nombre} (n={n})',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Tópico', fontsize=10)
    ax.set_ylabel('Palabra', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f'lda_heatmap_{nombre}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# DISTRIBUCION DE DOCUMENTOS POR TOPICO
# ======================================================

def graficar_distribucion_topicos(asig_df, nombre, n, dir_salida):
    conteo = asig_df['topico_dominante'].value_counts().sort_index()
    labels = [get_label(tid, nombre) for tid in conteo.index]
    palette = [PALETTE_TAB10[i % len(PALETTE_TAB10)] for i in range(len(conteo))]

    # Ordenar de mayor a menor
    orden = np.argsort(conteo.values)[::-1]
    labels_ord  = [labels[i]          for i in orden]
    valores_ord = [conteo.values[i]   for i in orden]
    colores_ord = [palette[i]         for i in orden]

    fig, ax = plt.subplots(figsize=(9, max(5, len(conteo) * 0.7)))
    bars = ax.barh(labels_ord[::-1], valores_ord[::-1],
                   color=colores_ord[::-1],
                   edgecolor='white', linewidth=0.5, alpha=0.88)

    for bar, val in zip(bars, valores_ord[::-1]):
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=9)

    ax.set_title(f'Documentos por tópico dominante — {nombre} (n={n})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Número de documentos', fontsize=10)
    ax.set_ylabel('Tópico', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f'lda_distribucion_topicos_{nombre}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# BOXPLOT DE PROBABILIDADES POR TOPICO
# ======================================================

def graficar_probabilidades_topico(asig_df, nombre, n, dir_salida):
    filas = []
    for tid in range(n):
        col = f'prob_topico_{tid}'
        if col in asig_df.columns:
            for val in asig_df[col]:
                filas.append({'tópico': get_label(tid, nombre), 'probabilidad': val})
    df_long = pd.DataFrame(filas)
    if df_long.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.7)))
    sns.boxplot(data=df_long, y='tópico', x='probabilidad',
                palette='tab10', orient='h', ax=ax,
                linewidth=0.8, fliersize=2)
    ax.set_title(f'Distribución de probabilidades por tópico — {nombre}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Probabilidad de pertenencia', fontsize=10)
    ax.set_ylabel('Tópico', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f'lda_probabilidades_{nombre}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# DEMOGRAFIA — estilo imagenes de referencia
# Eje Y = variable | Eje X = % | Leyenda "Temas Identificados"
# ======================================================

def graficar_demografia_lda(asig_df, variable, nombre, n, dir_salida):
    df = asig_df.copy()

    if variable == 'edad':
        df['_col'] = pd.to_numeric(df['edad'], errors='coerce')
        df = df.dropna(subset=['_col'])
        df['_col'] = df['_col'].astype(int).astype(str)
        col_var = '_col'
        label_eje = 'Edad'
    elif variable == 'genero':
        df['_col'] = df['genero']
        col_var = '_col'
        label_eje = 'Género'
    else:
        df['_col'] = df['lugar']
        col_var = '_col'
        label_eje = 'Ciudad de Origen'

    if col_var not in df.columns or df[col_var].isna().all():
        return

    tabla     = df.groupby([col_var, 'topico_dominante']).size().unstack(fill_value=0)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    ids_topico = sorted(tabla_pct.columns.tolist())
    palette    = sns.color_palette('tab10', n_colors=len(ids_topico))
    color_map  = {tid: palette[i] for i, tid in enumerate(ids_topico)}

    n_grupos = len(tabla_pct)
    fig, ax  = plt.subplots(figsize=(12, max(4, n_grupos * 1.1 + 1.5)))

    lefts  = np.zeros(n_grupos)
    grupos = tabla_pct.index.tolist()

    for tid in ids_topico:
        vals = tabla_pct[tid].values if tid in tabla_pct.columns else np.zeros(n_grupos)
        ax.barh(grupos, vals, left=lefts,
                color=color_map[tid],
                edgecolor='white', linewidth=0.6,
                label=label_corto_lda(tid, nombre))
        lefts += vals

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.set_xlabel('Porcentaje de participación', fontsize=11)
    ax.set_ylabel(label_eje, fontsize=11)
    ax.set_title(f'Distribución de Temas por {label_eje} — {nombre} (n={n})',
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(title='Temas Identificados',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8.5, title_fontsize=10,
              framealpha=0.95, edgecolor='#cccccc')
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f'demo_{variable}_{nombre}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# FRECUENCIAS DE NGRAM POR TOPICO
# Barras horizontales con gradiente plasma, top N terminos
# Lee directamente del CSV lda_topicos_{nombre}_n{k}.csv
# ======================================================

def graficar_ngram_por_topico(topicos_df, nombre, n, dir_salida, top_n=15):
    cmap = plt.get_cmap('plasma')

    for tid in range(n):
        df_t = topicos_df[topicos_df['topico'] == f'Topico_{tid}'].copy()
        if df_t.empty:
            continue

        df_t = df_t.nlargest(top_n, 'peso').sort_values('peso', ascending=True)
        palabras = df_t['palabra'].tolist()
        valores  = df_t['peso'].tolist()

        n_bars  = len(palabras)
        colores = [cmap(0.2 + 0.6 * (i / max(n_bars - 1, 1))) for i in range(n_bars)]

        titulo_topico = get_label(tid, nombre)
        fig, ax = plt.subplots(figsize=(9, max(5, n_bars * 0.45 + 1.5)))
        bars = ax.barh(palabras, valores, color=colores,
                       edgecolor='white', linewidth=0.5)

        ax.set_title(f'Top {top_n} términos — T{tid}: {titulo_topico}\n({nombre})',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Peso en el tópico', fontsize=10)
        ax.set_ylabel('Término', fontsize=10)

        for bar, val in zip(bars, valores):
            ax.text(bar.get_width() + max(valores) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', ha='left', fontsize=8)

        ax.set_xlim(0, max(valores) * 1.18)
        ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='gray')
        ax.set_axisbelow(True)
        sns.despine()
        plt.tight_layout()

        fname = f'ngram_topico{tid}_{nombre}.png'
        plt.savefig(os.path.join(dir_salida, fname), dpi=150, bbox_inches='tight')
        plt.close()


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

ranking = pd.read_csv(os.path.join(DIR_LDA, 'ranking_lda.csv'))

# Cargar data_nlp para tener columnas demograficas
df_nlp = pd.read_csv(PATH_NLP)

for nombre, n in N_OPTIMO.items():
    dir_salida = os.path.join(DIR_OUT, CARPETA[nombre])
    print(f'\n{nombre} | n={n}')

    topicos_df = pd.read_csv(os.path.join(DIR_LDA, f'lda_topicos_{nombre}_n{n}.csv'))
    asig_df    = pd.read_csv(os.path.join(DIR_LDA, f'lda_asignaciones_{nombre}_n{n}.csv'))

    # Merge con metadatos demograficos si no los tiene
    for col in ['genero', 'lugar', 'edad']:
        if col not in asig_df.columns and col in df_nlp.columns:
            asig_df = asig_df.merge(
                df_nlp[['comentario_cleaned', col]].drop_duplicates(),
                on='comentario_cleaned', how='left'
            ) if 'comentario_cleaned' in asig_df.columns else asig_df

    # Evaluacion
    print('  Perplexity / coherence...')
    graficar_perplexity(ranking, nombre, dir_salida)
    graficar_coherence(ranking, nombre, dir_salida)

    # Heatmap
    print('  Heatmap...')
    graficar_heatmap(topicos_df, nombre, n, dir_salida)

    # Distribucion
    print('  Distribucion de topicos...')
    graficar_distribucion_topicos(asig_df, nombre, n, dir_salida)

    # Probabilidades
    print('  Probabilidades...')
    graficar_probabilidades_topico(asig_df, nombre, n, dir_salida)

    # Frecuencias ngram por topico
    print('  Frecuencias por topico...')
    graficar_ngram_por_topico(topicos_df, nombre, n, dir_salida, top_n=15)

    # Demografia — se guarda en subcarpeta demos/
    print('  Graficas demograficas...')
    dir_demos_lda = os.path.join(dir_salida, 'demos')
    for var in ['genero', 'lugar', 'edad']:
        graficar_demografia_lda(asig_df, var, nombre, n, dir_demos_lda)

    print(f'  Guardado en {dir_salida}')

print('\nGraficado LDA completado.')