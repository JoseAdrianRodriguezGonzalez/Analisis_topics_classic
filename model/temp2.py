'''
generar_top5_txt.py
-------------------
Extrae las top 5 palabras de cada archivo tfidf_*.csv
en data/analisis_clusters/ y genera un TXT por combinacion
ngrama+modelo, listo para copiar y pegar en el prompt del LLM.

Salida en: data/analisis_clusters/prompts/
  top5_bigramas_dbscan.txt
  top5_bigramas_jerarquico.txt
  top5_unigramas_jerarquico.txt
  top5_trigramas_jerarquico.txt
'''

import os
import re
import pandas as pd

DIR_IN  = '../data/analisis_clusters/'
DIR_OUT = '../data/analisis_clusters/prompts/'
TOP_N   = 5

os.makedirs(DIR_OUT, exist_ok=True)

# Detectar todas las combinaciones ngrama+modelo disponibles
patron = re.compile(r'^tfidf_(\w+)_(\w+)_c(\d+)\.csv$')

combinaciones = {}  # {(ngrama, modelo): [lista de archivos ordenados por cluster]}

for archivo in os.listdir(DIR_IN):
    m = patron.match(archivo)
    if m:
        ngrama  = m.group(1)
        modelo  = m.group(2)
        cluster = int(m.group(3))
        clave   = (ngrama, modelo)
        if clave not in combinaciones:
            combinaciones[clave] = []
        combinaciones[clave].append((cluster, archivo))

# Para cada combinacion generar un TXT
for (ngrama, modelo), archivos in sorted(combinaciones.items()):
    archivos_ord = sorted(archivos, key=lambda x: x[0])  # ordenar por numero de cluster

    lineas = []
    lineas.append(f'TOP {TOP_N} PALABRAS POR CLUSTER')
    lineas.append(f'Metodo: {ngrama.upper()} | {modelo.upper()}')
    lineas.append('=' * 50)

    for cluster_id, archivo in archivos_ord:
        path = os.path.join(DIR_IN, archivo)
        df   = pd.read_csv(path)

        top  = df.head(TOP_N)
        n_docs = int(df['n_docs_cluster'].iloc[0]) if 'n_docs_cluster' in df.columns else '?'

        lineas.append(f'\nCluster {cluster_id} ({n_docs} documentos):')
        lineas.append(f"  {'Rank':<5} {'Palabra':<25} {'Score':>10}  {'IDF local':>10}  {'Docs con término':>16}")
        lineas.append(f"  {'-'*5} {'-'*25} {'-'*10}  {'-'*10}  {'-'*16}")

        for _, fila in top.iterrows():
            lineas.append(
                f"  {int(fila['rank']):<5} "
                f"{str(fila['palabra']):<25} "
                f"{fila['score_centroide']:>10.6f}  "
                f"{fila['idf_local']:>10.6f}  "
                f"{int(fila['df_en_cluster']):>16}"
            )

    lineas.append('\n' + '=' * 50)

    nombre_out = f'top5_{ngrama}_{modelo}.txt'
    with open(os.path.join(DIR_OUT, nombre_out), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas))

    print(f'Generado: {nombre_out}  ({len(archivos_ord)} clusters)')

print(f'\nArchivos en: {DIR_OUT}')

