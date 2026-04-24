'''
clustering_visualizacion.py
---------------------------
Bloque 6 — Visualización de resultados del grid search de clustering.
Genera gráficas diagnósticas usando exclusivamente seaborn.

Itera sobre las cuatro fuentes producidas por clustering_pipeline.py
y genera el mismo conjunto de gráficas para cada una en su subcarpeta.

Estructura de entrada esperada en data/clustering/:
    embeddings/
        ranking_completo.csv
        mejores_modelos.csv
        etiquetas_mejores.json
    features/
        ranking_completo.csv
        mejores_modelos.csv
        etiquetas_mejores.json
    tfidf/
        (mismos archivos)
    yake/
        (mismos archivos)
    comparacion_fuentes.csv   <-- tabla comparativa entre fuentes

Estructura de salida en data/clustering/:
    embeddings/graficas/
        01_silhouette_por_modelo.png
        02_kmeans_elbow_silhouette.png
        03_jerarquico_heatmap.png
        04_distribucion_clusters.png
        05_comparacion_mejores.png
        06_scatter_clusters.png
    features/graficas/
        (mismas gráficas)
    tfidf/graficas/
        (mismas gráficas)
    yake/graficas/
        (mismas gráficas)
    graficas_globales/
        07_comparacion_entre_fuentes.png  <-- gráfica comparativa global

Uso desde main.py:
    from clustering.clustering_visualizacion import run_visualizacion
    run_visualizacion()
'''

import json
import logging
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ======================================================
# RUTAS BASE
# ======================================================

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / 'data' / 'clustering'

# Ruta a los embeddings para la gráfica de scatter (fuente embeddings)
PATH_EMBEDS_NPY = BASE_DIR / 'data' / 'features' / 'docs_with_topics.npy'

# Ruta para la gráfica comparativa entre fuentes
PATH_COMPARACION = DATA_DIR / 'comparacion_fuentes.csv'

# Nombres de fuentes disponibles — en el mismo orden que se generaron
FUENTES_DISPONIBLES = ['embeddings', 'features', 'tfidf', 'yake']

# ======================================================
# ESTILO Y PALETAS
# ======================================================

PALETTE_MODELOS = {
    'jerarquico': '#2E86AB',
    'kmeans'    : '#E84855',
    'hdbscan'   : '#3BB273',
}

PALETTE_FUENTES = {
    'embeddings': '#7B2D8B',
    'features'  : '#2E86AB',
    'tfidf'     : '#E84855',
    'yake'      : '#3BB273',
}

sns.set_theme(
    style='whitegrid',
    font_scale=1.1,
    rc={
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'figure.facecolor' : '#F5F5F5',
        'axes.facecolor'   : '#FAFAFA',
        'font.family'      : 'DejaVu Sans',
    }
)

DPI = 150


# ======================================================
# HELPERS
# ======================================================

def _guardar(fig: plt.Figure, dir_graficas: Path, nombre: str) -> None:
    '''
    Guarda una figura en la carpeta de gráficas de la fuente correspondiente.
    Recibe la carpeta como parámetro para evitar depender de una variable global.
    '''
    dir_graficas.mkdir(parents=True, exist_ok=True)
    ruta = dir_graficas / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info('Guardada: %s', ruta)


def _extraer_metodo(hiperparametros: str) -> str:
    match = re.search(r'metodo=(\w+)', hiperparametros)
    return match.group(1) if match else 'n/a'


def _extraer_k(hiperparametros: str) -> int:
    match = re.search(r'k=(\d+)', hiperparametros)
    return int(match.group(1)) if match else 0


def _cargar_datos_fuente(dir_fuente: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    '''
    Carga ranking_completo.csv, mejores_modelos.csv y etiquetas_mejores.json
    desde la subcarpeta de una fuente. Retorna None si algún archivo falta.
    '''
    path_ranking = dir_fuente / 'ranking_completo.csv'
    path_mejores = dir_fuente / 'mejores_modelos.csv'
    path_etiq    = dir_fuente / 'etiquetas_mejores.json'

    for path in [path_ranking, path_mejores, path_etiq]:
        if not path.exists():
            logger.warning('Archivo no encontrado: %s — fuente omitida', path)
            return None

    ranking  = pd.read_csv(path_ranking)
    mejores  = pd.read_csv(path_mejores)
    with open(path_etiq, encoding='utf-8') as f:
        etiquetas = json.load(f)

    logger.info(
        'Datos cargados desde %s: %d combinaciones | %d mejores | %d etiquetas',
        dir_fuente.name, len(ranking), len(mejores), len(etiquetas)
    )
    return ranking, mejores, etiquetas


# ======================================================
# GRÁFICA 01 — Silhouette por algoritmo
# ======================================================

def grafica_silhouette_por_modelo(
    ranking: pd.DataFrame,
    dir_graficas: Path,
    nombre_fuente: str,
) -> None:
    logger.info('[%s] Generando 01_silhouette_por_modelo...', nombre_fuente)
    df = ranking[ranking['silhouette'].notna()].copy()
    if df.empty:
        logger.warning('[%s] Sin datos de silhouette — gráfica omitida', nombre_fuente)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F5F5F5')

    modelos_presentes = df['modelo'].unique().tolist()
    palette_local     = {m: PALETTE_MODELOS.get(m, '#888888') for m in modelos_presentes}

    sns.boxplot(
        data=df, x='modelo', y='silhouette', hue='modelo',
        palette=palette_local, width=0.4, linewidth=1.5, fliersize=0, legend=False, ax=ax
    )
    sns.stripplot(
        data=df, x='modelo', y='silhouette', hue='modelo',
        palette=palette_local, size=6, alpha=0.55, jitter=True, legend=False, ax=ax
    )

    ax.axhline(0, color='#999999', linestyle='--', linewidth=1, label='Silhouette = 0')
    ax.set_title(
        f'Silhouette Score — Distribución por algoritmo\nFuente: {nombre_fuente}',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_xlabel('Algoritmo', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    _guardar(fig, dir_graficas, '01_silhouette_por_modelo.png')


# ======================================================
# GRÁFICA 02 — KMeans elbow + silhouette
# ======================================================

def grafica_kmeans_elbow_silhouette(
    ranking: pd.DataFrame,
    dir_graficas: Path,
    nombre_fuente: str,
) -> None:
    logger.info('[%s] Generando 02_kmeans_elbow_silhouette...', nombre_fuente)
    km = ranking[ranking['modelo'] == 'kmeans'].copy()
    if km.empty:
        logger.warning('[%s] Sin resultados KMeans — gráfica omitida', nombre_fuente)
        return

    km['k'] = km['hiperparametros'].apply(_extraer_k)
    km = km.sort_values('k')

    codo_k  = int(km['codo_k'].dropna().iloc[0]) if 'codo_k' in km.columns and km['codo_k'].notna().any() else None
    inercia = km['inercia'].values.astype(float)
    km['inercia_norm'] = (inercia - inercia.min()) / (inercia.max() - inercia.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F5F5F5')

    sns.lineplot(data=km, x='k', y='inercia_norm', ax=ax,
                 color=PALETTE_MODELOS['kmeans'], linewidth=2.5, marker='o')
    ax2 = ax.twinx()
    sns.lineplot(data=km, x='k', y='silhouette', ax=ax2,
                 color='#FF8C00', linewidth=2.5, linestyle='--', marker='s')

    if codo_k:
        ax.axvline(codo_k, color='#444444', linestyle=':', linewidth=1.8,
                   label=f'Codo k={codo_k}')

    ax.set_title(
        f'KMeans — Elbow + Silhouette por k\nFuente: {nombre_fuente}',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_xlabel('Número de clusters (k)', fontsize=11)
    ax.set_ylabel('Inercia normalizada', fontsize=11, color=PALETTE_MODELOS['kmeans'])
    ax2.set_ylabel('Silhouette Score', fontsize=11, color='#FF8C00')
    ax.set_xticks(km['k'].tolist())
    sns.despine(ax=ax, right=False)
    fig.tight_layout()
    _guardar(fig, dir_graficas, '02_kmeans_elbow_silhouette.png')


# ======================================================
# GRÁFICA 03 — Jerárquico heatmap silhouette por (método, k)
# ======================================================

def grafica_jerarquico_heatmap(
    ranking: pd.DataFrame,
    dir_graficas: Path,
    nombre_fuente: str,
) -> None:
    logger.info('[%s] Generando 03_jerarquico_heatmap...', nombre_fuente)
    jer = ranking[ranking['modelo'] == 'jerarquico'].copy()
    if jer.empty:
        logger.warning('[%s] Sin resultados jerárquico — gráfica omitida', nombre_fuente)
        return

    jer['metodo'] = jer['hiperparametros'].apply(_extraer_metodo)
    jer['k']      = jer['hiperparametros'].apply(_extraer_k)

    pivot = jer.pivot_table(index='metodo', columns='k', values='silhouette', aggfunc='mean')
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F5F5F5')
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt='.3f', cmap='Blues',
        linewidths=0.5, linecolor='#CCCCCC',
        cbar_kws={'label': 'Silhouette Score', 'shrink': 0.7},
        annot_kws={'size': 9}
    )
    ax.set_title(
        f'Clustering Jerárquico — Silhouette por (método, k)\nFuente: {nombre_fuente}',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_xlabel('Número de clusters (k)', fontsize=11)
    ax.set_ylabel('Método de enlace', fontsize=11)
    fig.tight_layout()
    _guardar(fig, dir_graficas, '03_jerarquico_heatmap.png')


# ======================================================
# GRÁFICA 04 — Distribución de tamaño de clusters
# ======================================================

def grafica_distribucion_clusters(
    etiquetas: dict,
    dir_graficas: Path,
    nombre_fuente: str,
) -> None:
    logger.info('[%s] Generando 04_distribucion_clusters...', nombre_fuente)
    if not etiquetas:
        logger.warning('[%s] Sin etiquetas — gráfica omitida', nombre_fuente)
        return

    n_modelos = len(etiquetas)
    fig, axes = plt.subplots(1, n_modelos, figsize=(5 * n_modelos, 5))
    fig.patch.set_facecolor('#F5F5F5')

    if n_modelos == 1:
        axes = [axes]

    for ax, (key, labels) in zip(axes, etiquetas.items()):
        modelo  = key.split('|')[0]
        color   = PALETTE_MODELOS.get(modelo, '#888888')
        arr     = np.array(labels)
        validos = arr[arr != -1]
        ids, counts = np.unique(validos, return_counts=True)

        df_bar = pd.DataFrame(
            {'cluster': [f'C{i}' for i in ids], 'n_docs': counts}
        ).sort_values('n_docs', ascending=True)

        sns.barplot(
            data=df_bar, y='cluster', x='n_docs', ax=ax,
            color=color, orient='h', edgecolor='white', linewidth=0.6
        )

        for bar, val in zip(ax.patches, df_bar['n_docs']):
            ax.text(
                bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=9
            )

        n_ruido = int((arr == -1).sum())
        titulo  = f'{modelo.upper()}\n{key.split("|")[1]}'
        if n_ruido:
            titulo += f'\nRuido: {n_ruido} docs'

        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xlabel('Documentos', fontsize=10)
        ax.set_ylabel('Cluster', fontsize=10)
        if len(counts) > 0:
            ax.set_xlim(0, counts.max() * 1.18)
        sns.despine(ax=ax)

    fig.suptitle(
        f'Tamaño de clusters — Mejores modelos | Fuente: {nombre_fuente}',
        fontsize=14, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    _guardar(fig, dir_graficas, '04_distribucion_clusters.png')


# ======================================================
# GRÁFICA 05 — Comparación directa mejores modelos por fuente
# ======================================================

def grafica_comparacion_mejores(
    mejores: pd.DataFrame,
    dir_graficas: Path,
    nombre_fuente: str,
) -> None:
    logger.info('[%s] Generando 05_comparacion_mejores...', nombre_fuente)
    if mejores.empty:
        logger.warning('[%s] Sin mejores modelos — gráfica omitida', nombre_fuente)
        return

    df_long = mejores[['modelo', 'score_ranking', 'silhouette']].melt(
        id_vars='modelo', var_name='metrica', value_name='valor'
    )
    df_long['metrica'] = df_long['metrica'].map(
        {'score_ranking': 'Score combinado', 'silhouette': 'Silhouette'}
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F5F5F5')

    sns.barplot(
        data=df_long, x='modelo', y='valor', hue='metrica', ax=ax,
        palette=['#2E86AB', '#E84855'], edgecolor='white'
    )

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9
            )

    ax.set_title(
        f'Comparación directa — Mejores modelos por algoritmo\nFuente: {nombre_fuente}',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_xlabel('Algoritmo', fontsize=11)
    ax.set_ylabel('Valor de métrica', fontsize=11)
    ax.set_ylim(0, 1.15)
    sns.despine(ax=ax)
    fig.tight_layout()
    _guardar(fig, dir_graficas, '05_comparacion_mejores.png')


# ======================================================
# GRÁFICA 06 — Scatter 2D de clusters en espacio de la fuente
# ======================================================

def grafica_scatter_clusters(
    etiquetas: dict,
    dir_graficas: Path,
    nombre_fuente: str,
    X_2d: np.ndarray,
) -> None:
    '''
    Scatter plot 2D de los documentos coloreados por cluster.
    Recibe X_2d ya reducida para evitar recalcular UMAP por cada fuente.
    '''
    logger.info('[%s] Generando 06_scatter_clusters...', nombre_fuente)
    if not etiquetas or X_2d is None:
        logger.warning('[%s] Sin datos para scatter — gráfica omitida', nombre_fuente)
        return

    n_modelos = len(etiquetas)
    fig, axes = plt.subplots(1, n_modelos, figsize=(6 * n_modelos, 5))
    fig.patch.set_facecolor('#F5F5F5')

    if n_modelos == 1:
        axes = [axes]

    for ax, (key, labels_list) in zip(axes, etiquetas.items()):
        modelo = key.split('|')[0]
        labels = np.array(labels_list)

        # Verificar que las etiquetas tengan el mismo número de documentos que X_2d
        if len(labels) != len(X_2d):
            logger.warning(
                '[%s] Longitud de etiquetas (%d) no coincide con X_2d (%d) — panel omitido',
                nombre_fuente, len(labels), len(X_2d)
            )
            ax.set_visible(False)
            continue

        df_plot = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1], 'cluster': labels})
        ruido   = df_plot[df_plot['cluster'] == -1]
        validos = df_plot[df_plot['cluster'] != -1]

        # Convertir cluster a string para que seaborn use colores categóricos
        validos = validos.copy()
        validos['cluster'] = validos['cluster'].astype(str)

        sns.scatterplot(
            data=validos, x='x', y='y', hue='cluster',
            palette='tab10', ax=ax, s=35, alpha=0.85,
            edgecolor='white', linewidth=0.5
        )

        if not ruido.empty:
            ax.scatter(
                ruido['x'], ruido['y'],
                color='#B0B0B0', s=20, alpha=0.5,
                label='Ruido (-1)', edgecolors='none'
            )

        ax.set_title(
            f'Espacio 2D — {modelo.upper()} ({key.split("|")[1]})\nFuente: {nombre_fuente}',
            fontsize=12, fontweight='bold', pad=10
        )
        ax.set_xlabel('Componente 1', fontsize=10)
        ax.set_ylabel('Componente 2', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc='upper left',
            fontsize=9, title='Cluster', title_fontsize=10
        )
        sns.despine(ax=ax, left=True, bottom=True)

    fig.tight_layout()
    _guardar(fig, dir_graficas, '06_scatter_clusters.png')


# ======================================================
# GRÁFICA 07 — Comparación global entre fuentes (gráfica nueva)
# ======================================================

def grafica_comparacion_entre_fuentes(
    path_comparacion: Path,
    dir_graficas_global: Path,
) -> None:
    '''
    Lee comparacion_fuentes.csv y genera una gráfica de barras agrupadas
    donde el eje X son las fuentes, las barras son los algoritmos y la
    altura es el silhouette del mejor modelo de cada combinación.

    Permite ver de un vistazo qué fuente+algoritmo produce clusters
    más coherentes según silhouette.
    '''
    logger.info('Generando 07_comparacion_entre_fuentes...')

    if not path_comparacion.exists():
        logger.warning('comparacion_fuentes.csv no encontrado — gráfica 07 omitida')
        return

    df = pd.read_csv(path_comparacion)
    if df.empty:
        logger.warning('comparacion_fuentes.csv está vacío — gráfica 07 omitida')
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#F5F5F5')

    # Panel izquierdo: silhouette por fuente y algoritmo
    ax_sil = axes[0]
    sns.barplot(
        data=df, x='fuente', y='silhouette', hue='modelo',
        palette=PALETTE_MODELOS, ax=ax_sil,
        edgecolor='white', linewidth=0.6
    )
    for bar in ax_sil.patches:
        h = bar.get_height()
        if h > 0.02:
            ax_sil.text(
                bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8
            )
    ax_sil.set_title(
        'Silhouette por fuente y algoritmo',
        fontsize=13, fontweight='bold', pad=10
    )
    ax_sil.set_xlabel('Fuente de representación', fontsize=11)
    ax_sil.set_ylabel('Silhouette Score (mayor = mejor)', fontsize=11)
    ax_sil.set_ylim(0, 1.1)
    ax_sil.legend(title='Algoritmo', fontsize=9, title_fontsize=10)
    sns.despine(ax=ax_sil)

    # Panel derecho: número de clusters del mejor modelo por fuente y algoritmo
    ax_k = axes[1]
    sns.barplot(
        data=df, x='fuente', y='n_clusters', hue='modelo',
        palette=PALETTE_MODELOS, ax=ax_k,
        edgecolor='white', linewidth=0.6
    )
    for bar in ax_k.patches:
        h = bar.get_height()
        if h >= 1:
            ax_k.text(
                bar.get_x() + bar.get_width() / 2, h + 0.05,
                str(int(h)), ha='center', va='bottom', fontsize=8
            )
    ax_k.set_title(
        'Número de clusters óptimo por fuente y algoritmo',
        fontsize=13, fontweight='bold', pad=10
    )
    ax_k.set_xlabel('Fuente de representación', fontsize=11)
    ax_k.set_ylabel('Número de clusters (k)', fontsize=11)
    ax_k.legend(title='Algoritmo', fontsize=9, title_fontsize=10)
    sns.despine(ax=ax_k)

    fig.suptitle(
        'Comparación entre fuentes de representación vectorial',
        fontsize=15, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    _guardar(fig, dir_graficas_global, '07_comparacion_entre_fuentes.png')


# ======================================================
# REDUCCION 2D por fuente (para scatter)
# ======================================================

def _reducir_a_2d_para_scatter(
    nombre_fuente: str,
    dir_fuente: Path,
) -> np.ndarray | None:
    '''
    Carga proyeccion_2d.npy guardado por clustering_pipeline.py durante
    el procesamiento de cada fuente.

    Usar el mismo espacio 2D del pipeline garantiza que los puntos del
    scatter corresponden exactamente al espacio donde se midió el
    silhouette y se asignaron los clusters — no es una re-proyección.
    '''
    path_npy = dir_fuente / 'proyeccion_2d.npy'

    if not path_npy.exists():
        logger.warning(
            '[%s] proyeccion_2d.npy no encontrado en %s — scatter omitido. '
            'Vuelve a correr clustering_pipeline para regenerarlo.',
            nombre_fuente, dir_fuente
        )
        return None

    try:
        X_2d = np.load(path_npy)
        logger.info('[%s] Proyección 2D cargada desde %s  shape=%s', nombre_fuente, path_npy, X_2d.shape)
        return X_2d
    except Exception as error:
        logger.error('[%s] Error al cargar proyeccion_2d.npy: %s', nombre_fuente, error)
        return None


# ======================================================
# PIPELINE POR FUENTE
# ======================================================

def _visualizar_fuente(nombre_fuente: str, dir_fuente: Path) -> None:
    '''
    Ejecuta el pipeline completo de visualización para una fuente específica.
    Carga sus datos, genera las 6 gráficas y las guarda en su subcarpeta.
    '''
    logger.info('=== Visualizando fuente: %s ===', nombre_fuente)

    datos = _cargar_datos_fuente(dir_fuente)
    if datos is None:
        return

    ranking, mejores, etiquetas = datos
    dir_graficas = dir_fuente / 'graficas'

    # Generar proyección 2D para scatter (solo se calcula una vez por fuente)
    X_2d = _reducir_a_2d_para_scatter(nombre_fuente, dir_fuente)

    grafica_silhouette_por_modelo(ranking, dir_graficas, nombre_fuente)
    grafica_kmeans_elbow_silhouette(ranking, dir_graficas, nombre_fuente)
    grafica_jerarquico_heatmap(ranking, dir_graficas, nombre_fuente)
    grafica_distribucion_clusters(etiquetas, dir_graficas, nombre_fuente)
    grafica_comparacion_mejores(mejores, dir_graficas, nombre_fuente)
    grafica_scatter_clusters(etiquetas, dir_graficas, nombre_fuente, X_2d)

    logger.info('Fuente %s completada. Gráficas en: %s', nombre_fuente, dir_graficas)


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def run_visualizacion(
    fuentes: list[str] = None,
    generar_comparacion_global: bool = True,
) -> None:
    '''
    Genera las gráficas de diagnóstico del clustering para cada fuente
    y opcionalmente la gráfica comparativa global entre fuentes.

    Parámetros:
        fuentes                    -- lista de fuentes a procesar; si es None
                                      se procesan todas las disponibles
        generar_comparacion_global -- si True genera 07_comparacion_entre_fuentes.png
    '''
    fuentes_a_procesar = fuentes if fuentes is not None else FUENTES_DISPONIBLES

    # Gráficas individuales por fuente
    for nombre_fuente in fuentes_a_procesar:
        dir_fuente = DATA_DIR / nombre_fuente
        if not dir_fuente.exists():
            logger.info('Subcarpeta %s no existe — fuente omitida', dir_fuente)
            continue
        _visualizar_fuente(nombre_fuente, dir_fuente)

    # Gráfica comparativa global entre todas las fuentes
    if generar_comparacion_global:
        dir_graficas_global = DATA_DIR / 'graficas_globales'
        grafica_comparacion_entre_fuentes(PATH_COMPARACION, dir_graficas_global)

    logger.info('Visualización multi-fuente completada.')


if __name__ == '__main__':
    run_visualizacion()