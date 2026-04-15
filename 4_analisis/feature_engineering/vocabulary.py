'''
vocabulary.py
--------------
Genera vocabulario de n-gramas (unigramas, bigramas, trigramas) desde clean.csv
de Huatulco. Se ejecuta ANTES del feature_engineering para preparar los rankings
que necesitan las features de keywords.

El función principal genera 3 CSVs de rankings en data/processed/:
    -- rankings_unigrams.csv   (unigramas, 1-gramas)
    -- rankings_bigrams.csv    (bigramas, 2-gramas)
    -- rankings_trigrams.csv   (trigramas, 3-gramas)

Cada CSV contiene:
    -- ngram            : el n-grama como string
    -- total_frequency  : cantidad de veces que aparece en el corpus
    -- relative_frequency : proporcion respecto al total de n-gramas

Uso desde main.py:
    from feature_engineering.vocabulary import build_vocabulary_from_clean
    build_vocabulary_from_clean()

Uso directo:
    python vocabulary.py
'''

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.util import ngrams

# Rutas base
BASE_DIR      = Path(__file__).resolve().parents[1]
DATA_DIR      = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# Entrada
PATH_CLEAN = DATA_DIR / 'data_spanish' / 'clean.csv'

# Salidas
PATH_VOCAB_UNI = PROCESSED_DIR / 'rankings_unigrams.csv'
PATH_VOCAB_BI = PROCESSED_DIR / 'rankings_bigrams.csv'
PATH_VOCAB_TRI = PROCESSED_DIR / 'rankings_trigrams.csv'

logger = logging.getLogger(__name__)


def _build_ngrams_and_frequency(tokenized_texts: list[str], n: int) -> pd.DataFrame:
    '''
    Genera n-gramas a partir de una lista de textos ya tokenizados.

    Parametros:
        tokenized_texts -- lista de strings donde las palabras están separadas por espacios
        n               -- tamaño del n-grama (1 para unigramas, 2 para bigramas, etc.)

    Retorna:
        DataFrame con columnas: ngram, total_frequency, relative_frequency
        Ordenado por frecuencia descendente
    '''
    all_ngrams = []

    for text in tokenized_texts:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            continue

        tokens = text.split()

        # Filtrar tokens muy cortos (menos de 2 caracteres)
        tokens = [token for token in tokens if len(token) >= 2]

        if len(tokens) >= n:
            n_grams = list(ngrams(tokens, n))
            n_grams_str = [' '.join(gram) for gram in n_grams]
            all_ngrams.extend(n_grams_str)

    if not all_ngrams:
        logger.warning('No se generaron n-gramas para n=%d', n)
        return pd.DataFrame(columns=['ngram', 'total_frequency', 'relative_frequency'])

    frequency_count = Counter(all_ngrams)
    total_ngrams = sum(frequency_count.values())

    results = []
    for ngram, count in frequency_count.most_common():
        relative_frequency = count / total_ngrams if total_ngrams > 0 else 0
        results.append((ngram, count, relative_frequency))

    df = pd.DataFrame(
        results,
        columns=['ngram', 'total_frequency', 'relative_frequency']
    )
    return df.round({'relative_frequency': 6})


def _ensure_processed_dir():
    '''Crea la carpeta data/processed/ si no existe'''
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_vocabulary_from_clean(
    input_path: str | Path = PATH_CLEAN,
    output_uni: str | Path = PATH_VOCAB_UNI,
    output_bi: str | Path = PATH_VOCAB_BI,
    output_tri: str | Path = PATH_VOCAB_TRI,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Genera vocabulario de n-gramas desde clean.csv y exporta rankings a CSVs.

    Este es el paso previo a run_feature_pipeline() que prepara el vocabulario
    desde datos limpios de Huatulco.

    Parametros:
        input_path  -- ruta a clean.csv (defecto: data/data_spanish/clean.csv)
        output_uni  -- ruta salida unigramas (defecto: data/processed/rankings_unigrams.csv)
        output_bi   -- ruta salida bigramas (defecto: data/processed/rankings_bigrams.csv)
        output_tri  -- ruta salida trigramas (defecto: data/processed/rankings_trigrams.csv)

    Retorna:
        Tupla (df_unigrams, df_bigrams, df_trigrams)
    '''
    input_path = Path(input_path)
    output_uni = Path(output_uni)
    output_bi = Path(output_bi)
    output_tri = Path(output_tri)

    # Crear carpeta de salida
    _ensure_processed_dir()

    # Leer clean.csv
    if not input_path.exists():
        logger.error('Archivo no encontrado: %s', input_path)
        raise FileNotFoundError(f'No existe {input_path}')

    logger.info('Leyendo %s...', input_path)
    df = pd.read_csv(input_path)

    if 'comentario_clean' not in df.columns:
        logger.error('Columna "comentario_clean" no encontrada en %s', input_path)
        raise ValueError('clean.csv debe contener columna "comentario_clean"')

    texts = df['comentario_clean'].tolist()
    logger.info('Corpus cargado: %d comentarios', len(texts))

    # Generar n-gramas
    logger.info('Generando unigramas...')
    df_unigrams = _build_ngrams_and_frequency(texts, n=1)

    logger.info('Generando bigramas...')
    df_bigrams = _build_ngrams_and_frequency(texts, n=2)

    logger.info('Generando trigramas...')
    df_trigrams = _build_ngrams_and_frequency(texts, n=3)

    # Exportar
    logger.info('Exportando a %s...', output_uni)
    df_unigrams.to_csv(output_uni, index=False)

    logger.info('Exportando a %s...', output_bi)
    df_bigrams.to_csv(output_bi, index=False)

    logger.info('Exportando a %s...', output_tri)
    df_trigrams.to_csv(output_tri, index=False)

    logger.info('✓ Vocabulario generado exitosamente')
    logger.info(
        '  Unigramas: %d términos | Bigramas: %d términos | Trigramas: %d términos',
        len(df_unigrams),
        len(df_bigrams),
        len(df_trigrams),
    )

    return df_unigrams, df_bigrams, df_trigrams


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
    )
    build_vocabulary_from_clean()
