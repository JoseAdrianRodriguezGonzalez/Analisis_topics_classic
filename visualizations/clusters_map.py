import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# Directorio/Rutas
sns.set_theme(style="whitegrid", context="talk")

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_PADRE = os.path.dirname(DIR_ACTUAL)
DIR_DATOS = os.path.join(DIR_PADRE, "data", "analisis_clusters")
DIR_LLM = os.path.join(DIR_PADRE, "llm_model")  # Nueva carpeta de IA

print("Iniciando generación automática de Mapas de Clústeres (PCA y t-SNE)...")

# Función para generar los mapas
def generar_mapas_para_modelo(archivo_datos, archivo_llm, prefijo_salida, titulo_base):
    ruta_datos = os.path.join(DIR_DATOS, archivo_datos)
    ruta_llm = os.path.join(DIR_LLM, archivo_llm)

    print(f"\nProcesando: {prefijo_salida.replace('_', ' ').title()}")

    # 1. Cargar los datos
    try:
        df = pd.read_csv(ruta_datos)
        df = df.dropna(subset=['comentario_cleaned']).reset_index(drop=True)

        df_llm = pd.read_csv(ruta_llm)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    # 2. Mapear los números de clúster con los nombres
    col_llm = 'Cluster' if 'Cluster' in df_llm.columns else 'cluster'

    diccionario_nombres = {}
    for _, row in df_llm.iterrows():
        etiquetas = str(row['Etiquetas_Conceptuales']).split(',')
        nombre_corto = etiquetas[0].strip().title() if len(etiquetas) > 0 else f"Tema {row[col_llm]}"
        diccionario_nombres[row[col_llm]] = f"C{row[col_llm]}: {nombre_corto}"

    df['Nombre_Tema'] = df['cluster'].map(diccionario_nombres)
    df['Nombre_Tema'] = df['Nombre_Tema'].fillna("Desconocido")

    # 3. Vectorización TF-IDF
    print("Vectorizando texto...")
    tfidf = TfidfVectorizer(max_features=500)
    matriz_tfidf = tfidf.fit_transform(df['comentario_cleaned'])
    matriz_densa = matriz_tfidf.toarray()

    # Función interna para dibujar
    def dibujar_mapa(coordenadas, metodo, nombre_archivo):
        df_plot = pd.DataFrame({
            'Eje X': coordenadas[:, 0],
            'Eje Y': coordenadas[:, 1],
            'Tema': df['Nombre_Tema']
        })
        # Ordenar alfabéticamente
        df_plot = df_plot.sort_values('Tema')

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='Eje X', y='Eje Y',
            hue='Tema',
            palette=sns.color_palette("husl", len(df_plot['Tema'].unique())),  # Husl maneja mejor paletas grandes
            data=df_plot,
            alpha=0.8,
            s=250,
            edgecolor='white',
            linewidth=1.5
        )

        plt.title(f"{titulo_base} ({metodo})", fontsize=22, fontweight='bold', pad=20)
        plt.xlabel("Dimensión Semántica X", fontsize=14)
        plt.ylabel("Dimensión Semántica Y", fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Temas Identificados (IA)", fontsize=12,
                   title_fontsize=14)

        plt.tight_layout()
        ruta_salida = os.path.join(DIR_ACTUAL, nombre_archivo)
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        plt.close()

    # 4. APLICAR PCA
    print("Calculando PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_resultados = pca.fit_transform(matriz_densa)
    # Se quitó el "3_" del nombre del archivo
    dibujar_mapa(pca_resultados, "PCA", f"mapa_pca_{prefijo_salida}.png")

    # 5. APLICAR t-SNE
    print("Calculando t-SNE...")
    # Ajuste dinámico de perplexity para evitar errores si un cluster tiene muy poquitos datos
    perplexity_dinamico = min(15, max(5, len(df) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_dinamico)
    tsne_resultados = tsne.fit_transform(matriz_densa)
    # Se quitó el "3_" del nombre del archivo
    dibujar_mapa(tsne_resultados, "t-SNE", f"mapa_tsne_{prefijo_salida}.png")

    print(f"Mapas guardados para {prefijo_salida}")

# Ejecución
if __name__ == "__main__":
    # 1. Unigramas
    generar_mapas_para_modelo(
        "representativos_unigramas_jerarquico.csv",
        "llm_unigramas_jerarquico.csv",
        "unigramas_jerarquico",
        "Distribución de Temas: Unigramas"
    )

    # 2. Bigramas Jerárquico
    generar_mapas_para_modelo(
        "representativos_bigramas_jerarquico.csv",
        "llm_bigramas_jerarquico.csv",
        "bigramas_jerarquico",
        "Distribución de Temas: Bigramas"
    )

    # 3. Bigramas DBSCAN
    generar_mapas_para_modelo(
        "representativos_bigramas_dbscan.csv",
        "llm_bigramas_dbscan.csv",
        "bigramas_dbscan",
        "Distribución de Temas: Bigramas (DBSCAN)"
    )

    # 4. Trigramas
    generar_mapas_para_modelo(
        "representativos_trigramas_jerarquico.csv",
        "llm_trigramas_jerarquico.csv",
        "trigramas_jerarquico",
        "Distribución de Temas: Trigramas"
    )