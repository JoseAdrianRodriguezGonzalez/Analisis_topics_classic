import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración
sns.set_theme(style="whitegrid", context="talk")

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_PADRE = os.path.dirname(DIR_ACTUAL)

# Ruta de datos y resultados del análisis cualitativo
ruta_datos = os.path.join(DIR_PADRE, "data", "analisis_clusters", "representativos_unigramas_jerarquico.csv") #Puede cambiarse a cualquier otro modelo
ruta_llm = os.path.join(DIR_PADRE, "llm_model", "llm_unigramas_jerarquico.csv")

print("Iniciando procesamiento de visualizaciones demográficas...")

# 1. Carga de información
try:
    df = pd.read_csv(ruta_datos)
    df_analisis = pd.read_csv(ruta_llm)
    print("Datos cargados correctamente.")
except Exception as e:
    print(f"Error al cargar archivos: {e}")
    exit()

# 2. Mapeo de etiquetas generadas por el modelo de lenguaje
diccionario_temas = {}
for _, fila in df_analisis.iterrows():
    etiquetas = str(fila['Etiquetas_Conceptuales']).split(',')
    nombre_principal = etiquetas[0].strip().title() if len(etiquetas) > 0 else f"Tema {fila['Cluster']}"
    diccionario_temas[fila['Cluster']] = f"C{fila['Cluster']}: {nombre_principal}"

df['Tema_Identificado'] = df['cluster'].map(diccionario_temas)

#Procesamiento de variables
conteo_lugares = df['lugar'].value_counts()
principales = conteo_lugares[conteo_lugares >= 3].index
df['ubicacion_agrupada'] = df['lugar'].apply(lambda x: x if x in principales else 'Otros').str.title()

#Rangos de edad
def crear_rangos_edad(edad):
    if edad <= 19:
        return "17-19 años"
    elif edad <= 22:
        return "20-22 años"
    else:
        return "23+ años"

df['rango_edad'] = df['edad'].apply(crear_rangos_edad)


#Función de visualización
def generar_grafica_barras_apiladas(columna, titulo, nombre_archivo, texto_eje_y):
    print(f"Generando: {nombre_archivo}")

    # Crear tabla de frecuencias relativas
    tabla_cruzada = pd.crosstab(df[columna], df['Tema_Identificado'], normalize='index') * 100

    plt.figure(figsize=(16, 9))
    tabla_cruzada.plot(kind='barh', stacked=True, colormap='tab20', ax=plt.gca(), edgecolor='black', linewidth=0.8)

    plt.title(titulo, fontsize=24, fontweight='bold', pad=25)
    plt.xlabel("Porcentaje de Participación (%)", fontsize=18)
    plt.ylabel(texto_eje_y, fontsize=18)

    plt.gca().xaxis.set_major_formatter('{x:.0f}%')
    plt.legend(title="Temas del Análisis", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ACTUAL, nombre_archivo), dpi=300, bbox_inches='tight')
    plt.close()


#Ejecución
generar_grafica_barras_apiladas('genero', 'Distribución de Temas por Género', 'demografico_genero.png', 'Género')
generar_grafica_barras_apiladas('ubicacion_agrupada', 'Distribución de Temas por Procedencia',
                                'demografico_procedencia.png', 'Ciudad de Origen')
generar_grafica_barras_apiladas('rango_edad', 'Distribución de Temas por Rango de Edad', 'demografico_edad.png',
                                'Rango de Edad')
