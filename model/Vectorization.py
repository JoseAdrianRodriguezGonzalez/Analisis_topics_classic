'''
Aqui se ahce la vectorizacion de BoW y TF-IDF

A manera de funciones manuales
'''


import pandas as pd
import numpy as np
import os

#Funciones
def ImportarDatos(datos, vocabulario, dir_datos):
    '''
    datos = archivo con los comentarios limpios
    vocabulario = archivo con los ngrams
    '''


    path_datos = dir_datos + datos
    path_vocabulario = dir_datos + vocabulario


    dataset = pd.read_csv(path_datos)
    ngramas = pd.read_csv(path_vocabulario)

    return dataset, ngramas

def generar_ngrams(palabras, n):
    """
    Dado una lista de palabras y n, devuelve lista de n-gramas como strings
    Ejemplo: ["a","b","c"], n=2  ["a b", "b c"]
    """
    ngrams = []
    for i in range(len(palabras) - n + 1):
        ngram = " ".join(palabras[i:i+n])
        ngrams.append(ngram)
    return ngrams

def calcular_BoW(corpus, vocabulario, n=1):
    vocab_index = {palabra: idx for idx, palabra in enumerate(vocabulario)}  
    BoW = np.zeros((len(corpus), len(vocabulario)), dtype=int)

    comentarios_vacios = []
    for i, documento in enumerate(corpus):
        if pd.isna(documento):
            comentarios_vacios.append(i)
            #Borra esa fila de la BoW para evitar una fila llena de ceros
            continue
        palabras = documento.split()
        tokens = generar_ngrams(palabras, n)  # genera los n-gramas del documento
        for token in tokens:
            if token in vocab_index:  # búsqueda en dict, no en lista
                indice = vocab_index[token]
                BoW[i, indice] += 1

    #Eliminar filas donde todo es 0

    mask = BoW.sum(axis=1) > 0
    BoW = BoW[mask]  # solo filas sin ceros



    return BoW, comentarios_vacios, mask

def datos_sin_palabras(indice, path_datos, datos):
    '''
    Se crea un dataset que guarda los datos de las personas que palabras en sus comentarios

    para despues saber que tipo de grupos, no comento nada
    '''
    path_datos = path_datos + datos

    #crear dataset vacio
    original = pd.read_csv(path_datos)
    dataset_sin_palabras = pd.DataFrame(columns=['id', 'edad', 'lugar', 'genero'])
    #Para un indice dado, copiar los datos de las columnas originales en el nuevo (excepto el comentario ya que no hay)
    for i in indice:
        #
        fila = original.loc[i, ['edad', 'lugar', 'genero']]
        #Agrega la fila al nuevo dataset pero datafraame no tiene append
        dataset_sin_palabras = pd.concat([dataset_sin_palabras, pd.DataFrame({'id': [i], 'edad': [fila['edad']], 'lugar': [fila['lugar']], 'genero': [fila['genero']]})], ignore_index=True)
    return dataset_sin_palabras

                
def calcular_tf(BoW):
    totales = BoW.sum(axis=1, keepdims=True)  # vector (N, 1)
    TF = np.where(totales > 0, BoW / totales, 0)
    return TF

def calcular_idf(BoW):
    df = np.sum(BoW > 0, axis=0)  # vector (|V|,) con document frequency de cada término
    IDF = np.log((BoW.shape[0] + 1) / (df + 1)) + 1
    return IDF


def calcular_tf_idf(TF, IDF):
    return TF * IDF

def normalizacion_l2(TF_IDF):
    normas = np.linalg.norm(TF_IDF, axis=1, keepdims=True)  # vector (N, 1)
    TF_IDF_normalizado = np.where(normas > 0, TF_IDF / normas, 0)
    return TF_IDF_normalizado





#Ejecucion 

archivos = {'unigramas': 'rankings_unigrams.csv', 
            'bigramas': 'rankings_bigrams.csv', 
            'trigramas': 'rankings_trigrams.csv'}

n_values = {'unigramas': 1, 'bigramas': 2, 'trigramas': 3}


for nombre_archivo, archivo in archivos.items():
    n = n_values[nombre_archivo]
    dataset, ngram = ImportarDatos('data_nlp.csv', archivo, '../data/processed/')

    corpus = dataset['comentario_cleaned'].tolist()

    vocabulario = ngram['ngram'].tolist() #Pasar a array de numpy

    BoW, indices_vacios, mask = calcular_BoW(corpus, vocabulario, n=n)

    datset_alineado = dataset[mask].reset_index(drop=True)

    corpus_vacio = datos_sin_palabras(indices_vacios, '../data/processed/', 'data_nlp.csv') #Dataset con los datos de las personas que no comentaron nada

    TF = calcular_tf(BoW)

    IDF = calcular_idf(BoW)

    TF_IDF = calcular_tf_idf(TF, IDF)

    TF_IDF_normalizado = normalizacion_l2(TF_IDF)

    #exportar la matriz como csv
    df_TF_IDF = pd.DataFrame(TF_IDF_normalizado, columns=vocabulario)
    df_TF_IDF.to_csv(f'../data/processed/TF_IDF_normalizado_{nombre_archivo}.csv', index=False, encoding='utf-8-sig')

    corpus_vacio.to_csv(f'../data/processed/corpus_vacio.csv', index=False, encoding='utf-8-sig')

#marca warning pero si funciona

