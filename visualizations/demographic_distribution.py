import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directorio/Ruta
sns.set_theme(style="whitegrid", context="talk")

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_PADRE = os.path.dirname(DIR_ACTUAL)
ruta_datos = os.path.join(DIR_PADRE, "data", "analisis_clusters", "representativos_unigramas_jerarquico.csv")
ruta_llm = os.path.join(DIR_PADRE, "llm_model", "llm_unigramas_jerarquico.csv")

print("Iniciando análisis demográfico...")

# 1. Cargar datos
try:
    df = pd.read_csv(ruta_datos)
    df = df.dropna(subset=['comentario_cleaned']).reset_index(drop=True)
    df_llm = pd.read_csv(ruta_llm)
    print("Datos cargados correctamente.")
except Exception as e:
    print(f"Error: {e}")
    exit()

#Mapear los temas
diccionario_nombres = {}
for _, row in df_llm.iterrows():
    etiquetas = str(row['Etiquetas_Conceptuales']).split(',')
    nombre_corto = etiquetas[0].strip().title() if len(etiquetas) > 0 else f"Tema {row['Cluster']}"
    diccionario_nombres[row['Cluster']] = f"C{row['Cluster']}: {nombre_corto}"

df['Nombre_Tema'] = df['cluster'].map(diccionario_nombres)

#Procesamiento
#Agrupar Lugares
conteo_lugares = df['lugar'].value_counts()
lugares_principales = conteo_lugares[conteo_lugares >= 3].index
df['lugar_agrupado'] = df['lugar'].apply(lambda x: x if x in lugares_principales else 'Otros')
df['lugar_agrupado'] = df['lugar_agrupado'].str.title()

#Agrupar Edades en Rangos
def categorizar_edad(edad):
    if edad <= 19: return "17-19 años"
    elif edad <= 22: return "20-22 años"
    else: return "23+ años"

df['rango_edad'] = df['edad'].apply(categorizar_edad)

#Función para dibujar
def dibujar_distribucion(columna_demografica, titulo, nombre_archivo, xlabel_texto):
    print(f"Gráfica para: {columna_demografica}")
    cruce = pd.crosstab(df[columna_demografica], df['Nombre_Tema'], normalize='index') * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    #Paleta de colores
    cruce.plot(kind='barh', stacked=True, colormap='tab20', ax=ax, edgecolor='black', linewidth=0.5)

    plt.title(titulo, fontsize=22, fontweight='bold', pad=20)
    plt.xlabel("Porcentaje de participación", fontsize=16)
    plt.ylabel(xlabel_texto, fontsize=16)

    ax.xaxis.set_major_formatter('{x:.0f}%')
    plt.legend(title="Temas Identificados", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)

    plt.tight_layout()
    ruta_salida = os.path.join(DIR_ACTUAL, nombre_archivo)
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()

#Ejecución
dibujar_distribucion('genero', 'Distribución de Temas por Género', 'distribucion_genero.png', 'Género')
dibujar_distribucion('lugar_agrupado', 'Distribución de Temas por Ciudad de Origen', 'distribucion_origen.png', 'Ciudad de Origen')
dibujar_distribucion('rango_edad', 'Distribución de Temas por Rango de Edad', 'distribucion_edad.png', 'Rango de Edad')
