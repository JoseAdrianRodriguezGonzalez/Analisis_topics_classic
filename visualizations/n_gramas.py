import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import os

#Directorio/Rutas
sns.set_theme(style="whitegrid", context="talk")
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_PADRE = os.path.dirname(DIR_ACTUAL)
DIR_DATOS = os.path.join(DIR_PADRE, "data", "analisis_clusters")

print("Análisis de frecuencias (N-gramas)...")

#Función para graficar
def graficar_ngramas_desde_archivo(nombre_archivo_csv, rango_n, titulo, nombre_salida, paleta):
    ruta_csv = os.path.join(DIR_DATOS, nombre_archivo_csv)
    print(f"\nProcesando: {nombre_archivo_csv}...")

    try:
        df = pd.read_csv(ruta_csv)
        df = df.dropna(subset=['comentario_cleaned'])
        textos = df['comentario_cleaned']

        # Extracción de n-gramas
        vectorizador = CountVectorizer(ngram_range=rango_n, max_features=20)
        matriz = vectorizador.fit_transform(textos)
        sum_palabras = matriz.sum(axis=0)

        frecuencias = [(word, sum_palabras[0, idx]) for word, idx in vectorizador.vocabulary_.items()]
        frecuencias = sorted(frecuencias, key=lambda x: x[1], reverse=True)

        df_freq = pd.DataFrame(frecuencias, columns=['Término', 'Frecuencia'])

        # Gráfica
        plt.figure(figsize=(14, 8))

        # Agregamos edgecolor para la definición
        sns.barplot(
            x='Frecuencia', y='Término',
            data=df_freq,
            palette=paleta,
            hue='Término',
            legend=False,
            edgecolor='black',
            linewidth=1.2
        )

        plt.title(titulo, fontsize=22, fontweight='bold', pad=20)
        plt.xlabel("Frecuencia de aparición", fontsize=16)
        plt.ylabel("", fontsize=14)

        # Hacer los números del eje X
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tight_layout()

        # Guardar
        ruta_salida_png = os.path.join(DIR_ACTUAL, nombre_salida)
        plt.savefig(ruta_salida_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Guardado exitosamente como: {nombre_salida}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {ruta_csv}")
    except Exception as e:
        print(f"Error procesando los datos: {e}")

#Ejecución
# 1. Unigramas
graficar_ngramas_desde_archivo(
    "representativos_unigramas_jerarquico.csv",
    (1, 1),
    "Top 20 Palabras Más Frecuentes (Modelo Jerárquico)",
    "frecuencias_unigramas_jerarquico.png",
    "viridis"
)

# 2. Bigramas (Jerárquico)
graficar_ngramas_desde_archivo(
    "representativos_bigramas_jerarquico.csv",
    (2, 2),
    "Top 20 Frases de 2 palabras (Modelo Jerárquico)",
    "frecuencias_bigramas_jerarquico.png",
    "magma"
)

# 3. Bigramas (DBSCAN)
graficar_ngramas_desde_archivo(
    "representativos_bigramas_dbscan.csv",
    (2, 2),
    "Top 20 Frases de 2 palabras (Modelo DBSCAN)",
    "frecuencias_bigramas_dbscan.png",
    "plasma"
)

# 4. Trigramas (Jerárquico)
graficar_ngramas_desde_archivo(
    "representativos_trigramas_jerarquico.csv",
    (3, 3),
    "Top 20 Frases de 3 palabras (Modelo Jerárquico)",
    "frecuencias_trigramas_jerarquico.png",
    "mako"
)

print("\nGráficas de N-gramas")