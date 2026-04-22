'''
cooccurrence_graph.py
---------------------
Bloque 8 — Co-ocurrencia de entidades y términos dentro de documentos.

Construye matrices y grafos de co-ocurrencia a dos niveles:
    1. Co-ocurrencia de entidades NER (qué entidades aparecen juntas)
    2. Co-ocurrencia de términos TF-IDF por tópico (qué palabras
       se asocian dentro de cada grupo temático)

Las co-ocurrencias se calculan a nivel de documento:
    Si dos entidades/términos aparecen en el mismo documento,
    se incrementa su co-ocurrencia en 1. El resultado es una
    matriz simétrica NxN.

Salidas:
    coocurrencia_entidades.csv     -- matriz en formato edge list
                                      (entidad_a, entidad_b, peso, pmi)
    coocurrencia_terminos.csv      -- co-ocurrencia de términos por tópico
    resumen_coocurrencia.csv       -- tabla de comunidades detectadas

Nota sobre PMI (Pointwise Mutual Information):
    PMI(a,b) = log2( P(a,b) / (P(a) * P(b)) )
    Un PMI positivo indica que dos términos co-ocurren más de lo
    esperado por azar. Se usa como peso normalizado del grafo.
'''

import json
import logging
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================
# RUTAS
# ======================================================



BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'analysis' / 'cooccurrence'

PATH_NER_GROUPS    = BASE_DIR/'data' / 'features'  / 'ner_groups.json'
PATH_DOCS_TOPICS   = BASE_DIR /'data' / 'results' / 'docs_with_topics.csv'
PATH_CLEAN_CSV     = BASE_DIR/'data' / 'data_lang' / 'data_spanish' / 'clean.csv'
PATH_VOCAB_UNI     = BASE_DIR /'data'  / 'processed' / 'rankings_unigrams.csv'

# Umbrales
MIN_DOCS_ENTIDAD       = 10   # entidades con al menos N documentos
MIN_COOCURRENCIA       = 5    # pares que co-ocurren al menos N veces
TOP_VOCAB_TERMINOS     = 200  # términos del vocabulario a incluir
MIN_COOC_TERMINOS      = 3    # mínimo de co-ocurrencias para términos


# ======================================================
# CO-OCURRENCIA DE ENTIDADES
# ======================================================

def _construir_indice_entidades(
    ner_grupos: list[dict],
    min_docs: int,
) -> dict[int, list[str]]:
    '''
    Construye un índice inverso: {indice_documento: [lista_de_entidades]}.
    Solo incluye entidades que superan el umbral de frecuencia min_docs.
    '''
    # Filtrar entidades poco frecuentes
    entidades_validas = {
        grupo['text']
        for grupo in ner_grupos
        if len(grupo['indices']) >= min_docs
    }
    logger.info(
        'Entidades válidas (>= %d docs): %d de %d',
        min_docs,
        len(entidades_validas),
        len(ner_grupos),
    )

    # Construir índice invertido
    indice: dict[int, list[str]] = {}
    for grupo in ner_grupos:
        if grupo['text'] not in entidades_validas:
            continue
        for idx in grupo['indices']:
            indice.setdefault(idx, []).append(grupo['text'])

    return indice


def calcular_coocurrencia_entidades(
    ner_grupos: list[dict],
    n_documentos_total: int,
) -> pd.DataFrame:
    '''
    Calcula la co-ocurrencia de pares de entidades y su PMI.

    Retorna DataFrame con columnas:
        entidad_a, entidad_b, co_ocurrencias, pmi, doc_freq_a, doc_freq_b
    Ordenado por co_ocurrencias descendente.
    '''
    indice = _construir_indice_entidades(ner_grupos, MIN_DOCS_ENTIDAD)

    # Frecuencia marginal de cada entidad (número de documentos)
    freq_marginal: dict[str, int] = {}
    for grupo in ner_grupos:
        if len(grupo['indices']) >= MIN_DOCS_ENTIDAD:
            freq_marginal[grupo['text']] = len(grupo['indices'])

    # Contar co-ocurrencias
    cooc: dict[tuple[str, str], int] = {}
    for doc_entidades in indice.values():
        entidades_unicas = list(set(doc_entidades))
        for ent_a, ent_b in combinations(sorted(entidades_unicas), 2):
            clave = (ent_a, ent_b)
            cooc[clave] = cooc.get(clave, 0) + 1

    logger.info('Pares de entidades calculados: %d pares', len(cooc))

    # Construir DataFrame y calcular PMI
    filas = []
    n = n_documentos_total

    for (ent_a, ent_b), conteo in cooc.items():
        if conteo < MIN_COOCURRENCIA:
            continue

        p_ab = conteo / n
        p_a  = freq_marginal.get(ent_a, 1) / n
        p_b  = freq_marginal.get(ent_b, 1) / n

        denominador = p_a * p_b
        pmi = math.log2(p_ab / denominador) if denominador > 0 and p_ab > 0 else 0.0

        filas.append({
            'entidad_a'      : ent_a,
            'entidad_b'      : ent_b,
            'co_ocurrencias' : conteo,
            'pmi'            : round(pmi, 4),
            'doc_freq_a'     : freq_marginal.get(ent_a, 0),
            'doc_freq_b'     : freq_marginal.get(ent_b, 0),
        })

    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values('co_ocurrencias', ascending=False).reset_index(drop=True)

    logger.info(
        'Co-ocurrencias de entidades (>= %d): %d pares',
        MIN_COOCURRENCIA,
        len(df),
    )
    return df


# ======================================================
# CO-OCURRENCIA DE TÉRMINOS POR TÓPICO
# ======================================================

def _cargar_vocabulario(path: Path, top_n: int) -> list[str]:
    '''
    Carga el vocabulario de unigramas y retorna los top_n términos
    más frecuentes.
    '''
    df = pd.read_csv(path)
    vocabulario = df['ngram'].head(top_n).tolist()
    logger.info('Vocabulario cargado: %d términos (top %d)', len(vocabulario), top_n)
    return vocabulario


def _tokenizar_documento(texto: str, vocab_set: set[str]) -> list[str]:
    '''
    Tokeniza un documento y filtra solo los tokens que pertenecen
    al vocabulario establecido.
    '''
    if not isinstance(texto, str) or not texto.strip():
        return []
    return [token for token in texto.split() if token in vocab_set]


def calcular_coocurrencia_terminos(
    df_corpus: pd.DataFrame,
    vocabulario: list[str],
) -> pd.DataFrame:
    '''
    Calcula la co-ocurrencia de pares de términos por tópico BERTopic.

    Para cada tópico válido (topic != -1), genera una edge list con
    los pares de términos que co-ocurren y su peso normalizado por PMI.

    Retorna DataFrame con columnas:
        topic, termino_a, termino_b, co_ocurrencias, pmi_dentro_topico
    '''
    if 'topic' not in df_corpus.columns:
        logger.warning('Columna topic no disponible. Saltando co-ocurrencia de términos.')
        return pd.DataFrame()

    vocab_set = set(vocabulario)
    topicos_validos = sorted(df_corpus[df_corpus['topic'] != -1]['topic'].unique())
    logger.info('Calculando co-ocurrencia de términos para %d tópicos...', len(topicos_validos))

    todas_filas = []

    for topico in topicos_validos:
        docs_topico = df_corpus[df_corpus['topic'] == topico]['comentario_clean'].tolist()
        n_docs_topico = len(docs_topico)

        # Frecuencia marginal de términos dentro del tópico
        freq_termino: dict[str, int] = {}
        cooc_topico: dict[tuple[str, str], int] = {}

        for doc in docs_topico:
            tokens = _tokenizar_documento(doc, vocab_set)
            tokens_unicos = list(set(tokens))
            for token in tokens_unicos:
                freq_termino[token] = freq_termino.get(token, 0) + 1
            for tok_a, tok_b in combinations(sorted(tokens_unicos), 2):
                clave = (tok_a, tok_b)
                cooc_topico[clave] = cooc_topico.get(clave, 0) + 1

        # Construir edge list del tópico
        for (tok_a, tok_b), conteo in cooc_topico.items():
            if conteo < MIN_COOC_TERMINOS:
                continue
            if freq_termino.get(tok_a, 0) < 2 or freq_termino.get(tok_b, 0) < 2:
                continue

            p_ab = conteo / n_docs_topico
            p_a  = freq_termino[tok_a] / n_docs_topico
            p_b  = freq_termino[tok_b] / n_docs_topico
            denominador = p_a * p_b
            pmi = math.log2(p_ab / denominador) if denominador > 0 and p_ab > 0 else 0.0

            todas_filas.append({
                'topic'                : topico,
                'termino_a'            : tok_a,
                'termino_b'            : tok_b,
                'co_ocurrencias'       : conteo,
                'pmi_dentro_topico'    : round(pmi, 4),
                'freq_a_en_topico'     : freq_termino[tok_a],
                'freq_b_en_topico'     : freq_termino[tok_b],
                'n_docs_topico'        : n_docs_topico,
            })

    df = pd.DataFrame(todas_filas)
    if not df.empty:
        df = df.sort_values(['topic', 'co_ocurrencias'], ascending=[True, False]).reset_index(drop=True)

    logger.info('Co-ocurrencias de términos: %d filas en %d tópicos', len(df), len(topicos_validos))
    return df


# ======================================================
# RESUMEN DE COMUNIDADES (DETECCIÓN SIMPLE POR COMPONENTES)
# ======================================================

def _detectar_comunidades_simples(
    df_cooc_entidades: pd.DataFrame,
    top_n_por_comunidad: int = 10,
) -> pd.DataFrame:
    '''
    Detección básica de comunidades usando componentes conectados
    del grafo de co-ocurrencia. No requiere librerías externas de grafos.

    Para cada componente conectado, lista las entidades con mayor
    centralidad de grado (suma de co-ocurrencias con sus vecinos).
    '''
    if df_cooc_entidades.empty:
        return pd.DataFrame()

    # Construir grafo como diccionario de adjacencia
    grafo: dict[str, set[str]] = {}
    pesos: dict[str, int] = {}

    for _, fila in df_cooc_entidades.iterrows():
        a, b = fila['entidad_a'], fila['entidad_b']
        grafo.setdefault(a, set()).add(b)
        grafo.setdefault(b, set()).add(a)
        pesos[a] = pesos.get(a, 0) + fila['co_ocurrencias']
        pesos[b] = pesos.get(b, 0) + fila['co_ocurrencias']

    # BFS para encontrar componentes conectados
    visitados: set[str] = set()
    comunidades: list[list[str]] = []

    for nodo_inicio in grafo:
        if nodo_inicio in visitados:
            continue
        componente: list[str] = []
        cola = [nodo_inicio]
        while cola:
            nodo = cola.pop()
            if nodo in visitados:
                continue
            visitados.add(nodo)
            componente.append(nodo)
            cola.extend(grafo.get(nodo, set()) - visitados)
        comunidades.append(componente)

    # Construir tabla de resumen
    filas = []
    for comunidad_id, componente in enumerate(comunidades):
        nodos_ordenados = sorted(componente, key=lambda x: pesos.get(x, 0), reverse=True)
        top_nodos = nodos_ordenados[:top_n_por_comunidad]
        filas.append({
            'comunidad_id'      : comunidad_id,
            'n_entidades'       : len(componente),
            'entidades_top'     : ', '.join(top_nodos),
            'entidad_central'   : top_nodos[0] if top_nodos else '',
            'grado_central'     : pesos.get(top_nodos[0], 0) if top_nodos else 0,
        })

    df = (
        pd.DataFrame(filas)
        .sort_values('n_entidades', ascending=False)
        .reset_index(drop=True)
    )

    logger.info('Comunidades detectadas: %d', len(df))
    return df


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def run_cooccurrence_graph(
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    '''
    Pipeline completo de construcción del grafo de co-ocurrencia.

    Retorna diccionario con tres DataFrames:
        cooc_entidades, cooc_terminos, comunidades
    '''
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('=== Iniciando análisis de co-ocurrencia ===')

    # Cargar datos
    if not PATH_NER_GROUPS.exists():
        logger.error('ner_groups.json no encontrado — pipeline abortado')
        return {}

    with open(PATH_NER_GROUPS, encoding='utf-8') as f:
        ner_grupos = json.load(f)

    df_corpus = pd.read_csv(PATH_DOCS_TOPICS)
    n_total   = len(df_corpus)
    logger.info('Corpus: %d documentos', n_total)

    # --- Co-ocurrencia de entidades ---
    logger.info('Calculando co-ocurrencia de entidades...')
    df_cooc_entidades = calcular_coocurrencia_entidades(ner_grupos, n_total)
    df_cooc_entidades.to_csv(
        output_dir / 'coocurrencia_entidades.csv',
        index=False, encoding='utf-8-sig',
    )

    # --- Co-ocurrencia de términos por tópico ---
    if PATH_VOCAB_UNI.exists() and PATH_CLEAN_CSV.exists():
        logger.info('Calculando co-ocurrencia de términos por tópico...')
        vocabulario = _cargar_vocabulario(PATH_VOCAB_UNI, TOP_VOCAB_TERMINOS)
        df_cooc_terminos = calcular_coocurrencia_terminos(df_corpus, vocabulario)
        df_cooc_terminos.to_csv(
            output_dir / 'coocurrencia_terminos.csv',
            index=False, encoding='utf-8-sig',
        )
    else:
        logger.warning(
            'Vocabulario o corpus limpio no encontrado — co-ocurrencia de términos omitida.'
        )
        df_cooc_terminos = pd.DataFrame()

    # --- Detección de comunidades ---
    logger.info('Detectando comunidades en el grafo de entidades...')
    df_comunidades = _detectar_comunidades_simples(df_cooc_entidades)
    df_comunidades.to_csv(
        output_dir / 'comunidades_entidades.csv',
        index=False, encoding='utf-8-sig',
    )

    logger.info('=== Co-ocurrencia completada. Archivos en: %s ===', output_dir)

    return {
        'cooc_entidades' : df_cooc_entidades,
        'cooc_terminos'  : df_cooc_terminos,
        'comunidades'    : df_comunidades,
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_cooccurrence_graph()
