import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud

"""
Función para generar una nube de palabras con la silueta de Salamanca.
"""
def nube_con_silueta(texto_limpio, ruta_imagen_mascara):
    mascara_salamanca = np.array(Image.open(ruta_imagen_mascara))

    #Configuración de la nube de palabras
    wordcloud = WordCloud(
        background_color="white", 
        mask=mascara_salamanca, 
        contour_width=2,         
        contour_color='darkred', 
        colormap='viridis', # Paleta de colores
        max_words=200
    ).generate(texto_limpio)

    #Gráfica de la nube de palabras
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()