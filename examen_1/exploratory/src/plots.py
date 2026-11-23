import matplotlib.pyplot as plt
# Colores para emociones
colores = ["red", "blue", "green", "orange", "purple", "brown"]
# Formas para bebidas
formas = ["o", "s", "^", "D"]

plt.figure(figsize=(8,6))

# Graficar bebidas como centroides de emociones (promedio de cada fila)
for i, bebida in enumerate(df.index):
    x = df.loc[bebida].dot(emotions_2d["PC1"])
    y = df.loc[bebida].dot(emotions_2d["PC2"])
    plt.scatter(x, y, marker=formas[i], s=200, label=bebida, edgecolors='k')

# Graficar emociones como puntos
for i, emotion in enumerate(emotions_2d.index):
    plt.scatter(emotions_2d["PC1"][i], emotions_2d["PC2"][i],
                color=colores[i], s=150, label=emotion, alpha=0.7)

plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Bebidas y emociones en el mismo plano (formas y colores)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
#plt.figure(figsize=(8,6))
#for i, bebida in enumerate(bebidas):
#    x, y = embedding_2d[i]
#    plt.scatter(x, y, label=bebida, s=100)  # tamaño 100 para resaltar
#    plt.text(x + 0.001, y + 0.001, bebida, fontsize=12)  # etiqueta cerca del punto

#plt.title("Visualización PCA de embeddings de bebidas")
#plt.xlabel("Componente principal 1")
#plt.ylabel("Componente principal 2")
#plt.grid(True)
#plt.legend()
#plt.show()
