import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import os

#Directorio/Rutas
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_SCRIPT)

#Datos de entrada
ruta_datos = os.path.join(DIR_RAIZ, "data", "analisis_clusters", "representativos_unigramas_jerarquico.csv")
#Imagen de silueta
ruta_imagen = os.path.join(DIR_SCRIPT, "image.png")
ruta_llm = os.path.join(DIR_RAIZ, "llm_model", "llm_unigramas_jerarquico.csv")

#Salida
ruta_salida = os.path.join(DIR_SCRIPT, "wordcloud_salamanca.png")

print("Generación de Nube de Palabras...")

#Texto limpio
try:
    df = pd.read_csv(ruta_datos)
    df = df.dropna(subset=['comentario_cleaned'])
    texto_completo = " ".join(df['comentario_cleaned'].tolist())
    print("Texto para la nube cargado.")

    df_llm = pd.read_csv(ruta_llm)
    print("Etiquetas LLM cargadas.")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    exit()

#Manejo de transparencia
try:
    img = Image.open(ruta_imagen)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        fondo_blanco = Image.new('RGB', img.size, (255, 255, 255))
        fondo_blanco.paste(img, (0, 0), img)
        img = fondo_blanco
    else:
        img = img.convert('RGB')

    img_gris = img.convert('L')
    mascara_bruta = np.array(img_gris)
    # Los píxeles claros se vuelven blanco puro, los oscuros negro puro
    mascara_limpia = np.where(mascara_bruta > 128, 255, 0).astype(np.uint8)
    print("Máscara de Salamanca procesada con éxito.")

except FileNotFoundError:
    print(f"Error: No se encontró la imagen en {ruta_imagen}.")
    exit()

# 3. Diseño estético de la Nube de Palabras
print("Generando la silueta con las palabras...")
wc = WordCloud(
    background_color="white",
    max_words=250,
    mask=mascara_limpia,
    contour_width=1.5,
    contour_color='black',
    colormap='inferno',
    random_state=42
)

wc.generate(texto_completo)

# 4. Renderizado Final en Alta Calidad
plt.figure(figsize=(12, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

# Título y subtítulo
plt.title("Percepción Estudiantil sobre Salamanca, Gto.", fontsize=24, fontweight='bold', pad=20)
plt.figtext(0.5, 0.05, "Nube de Palabras basada en 10 temas identificados por el LLM Qwen", ha="center", fontsize=12,
            color="gray")

plt.tight_layout()
plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
plt.close()

print(f"Nube guardada como: {ruta_salida}")