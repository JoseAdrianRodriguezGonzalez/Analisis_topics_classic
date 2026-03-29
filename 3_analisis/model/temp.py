#archivo temporal de diversos scripts pequenios


import pandas as pd
import os

dir = '../data/processed/'
input_path = '../data/processed/rankings_frequencies.xlsx'
output_folder = '../data/processed/'

dict_hojas = pd.read_excel(input_path, sheet_name=None)

for nombre_hoja, df in dict_hojas.items():
    nombre_archivo = f"rankings_{nombre_hoja.lower().replace(' ', '_')}.csv"
    ruta_salida = os.path.join(output_folder, nombre_archivo)
    
    df.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
    print(f'Procesada: {nombre_hoja} = {nombre_archivo}')

datos = input_path = '../data/processed/data_nlp.csv'

df = pd.read_csv(datos)

#checar ngramas

archivos = {'unigramas': (1, 'rankings_unigrams.csv'), 
            'bigramas': (2, 'rankings_bigrams.csv'), 
            'trigramas': (3, 'rankings_trigrams.csv')}

for nombre_archivo, archivo in archivos.items():
    n, nombre_archivo = archivo
    df_ngrams = pd.read_csv(os.path.join(dir, nombre_archivo))
    print(f"Procesada: {nombre_archivo} = {n}-gramas")

    #verificar si un ngram tiene una palabra de menos de 2 caracteres, que me diga el indice

    ngrams = df_ngrams['ngram'].tolist()
    for i, ngram in enumerate(ngrams):
        if pd.isna(ngram):
            continue
        palabras = ngram.split()
        for palabra in palabras:
            if len(palabra) <= 2:
                print(f"El ngram '{ngram}' en el archivo '{nombre_archivo}' tiene una palabra de 2 o menos caracteres: '{palabra}' (índice: {i})")