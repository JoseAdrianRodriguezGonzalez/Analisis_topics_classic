'''
features.py
-----------
Orquestador principal del bloque 3: Feature Engineering NLP.

Lee data_nlp.csv, ejecuta los cuatro modulos de features y exporta
un unico CSV consolidado (features_nlp.csv) listo para usar en
modelos predictivos o analisis estadistico.

Columnas del CSV de salida:
    -- Metadatos del documento (indice, edad, genero, lugar)
    -- Features de longitud  (text_features.py)
    -- Features de keywords  (keyword_features.py)
    -- Features POS          (pos_features.py)
    -- Features de entidades (entity_features.py)

Uso desde main.py:
    from feature_engineering.features import run_feature_pipeline
    run_feature_pipeline()

Uso directo:
    python features.py
'''

import logging
import sys
from pathlib import Path
import json
import pandas as pd
import os 
# Rutas base del proyecto
BASE_DIR      = Path(__file__).resolve().parents[1]
DATA_DIR      = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR  = DATA_DIR / 'features'

# Archivos de entrada - Lee directamente del preprocessing
PATH_DATA_CLEAN   = DATA_DIR / 'translations' / 'normalized_spanish.csv'
PATH_DATA_ANALYSIS = DATA_DIR / 'data_spanish' / 'analysis.csv'
PATH_VOCAB_UNI    = PROCESSED_DIR / 'rankings_unigrams.csv'
PATH_ANALYSIS_JSON = DATA_DIR / 'data_spanish' / 'analysis.json'

# Archivo de salida
PATH_FEATURES_OUT = FEATURES_DIR / 'features_nlp.csv'

# Columnas demograficas que se preservan del CSV original
DEMOGRAPHIC_COLUMNS = ['edad', 'genero', 'lugar']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def load_all_analysis_csv(data_dir):
    paths = [
        data_dir / 'data_spanish' / 'analysis.csv',
        data_dir / 'data_english' / 'analysis.csv',
        data_dir / 'data_mixed'   / 'analysis.csv',
    ]
    dfs = []
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["indice"])

    return df_all
def load_all_analysis_json(data_dir):
    paths = [
        data_dir / 'data_spanish' / 'analysis.json',
        data_dir / 'data_english' / 'analysis.json',
        data_dir / 'data_mixed'   / 'analysis.json',
    ]
    all_data = []
    for p in paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    seen = set()
    unique = []
    for row in all_data:
        idx = row["indice"]
        if idx not in seen:
            unique.append(row)
            seen.add(idx)
    return unique
def _load_corpus(clean_path: Path, analysis_path: Path | None = None) -> pd.DataFrame:
    '''
    Carga los comentarios limpios desde clean.csv y opcionalmente mergeea
    con analysis.csv para obtener metadatos adicionales (estrellas, etc).
    
    Retorna el DataFrame con índice y comentarios_cleaned alineados.
    '''
    # Cargar comentarios limpios
    df_clean = pd.read_csv(clean_path)
    
    # Renombrar columna si es necesario
    if 'comentario_clean' in df_clean.columns:
        df_clean = df_clean.rename(columns={'comentario_clean': 'comentario_cleaned'})
    
    # Opcionalmente mergear con análisis para obtener metadatos
    if analysis_path is not None and Path(analysis_path).exists():
        df_analysis = pd.read_csv(analysis_path)
        df_clean = df_clean.merge(df_analysis, on='indice', how='left')
    
    n_original = len(df_clean)
    mask = df_clean['comentario_cleaned'].notna() & (df_clean['comentario_cleaned'].str.strip() != '')
    df_valid = df_clean[mask].reset_index(drop=False)
    
    logger.info(
        'Corpus cargado: %d documentos validos de %d totales',
        len(df_valid),
        n_original,
    )
    return df_valid


def run_feature_pipeline(
    data_clean_path: str | Path         = PATH_DATA_CLEAN,
    data_analysis_path: str | Path      = PATH_DATA_ANALYSIS,
    vocab_unigrams_path: str | Path     = PATH_VOCAB_UNI,
    analysis_json_path: str | Path      = PATH_ANALYSIS_JSON,
    output_path: str | Path             = PATH_FEATURES_OUT,
) -> pd.DataFrame:
    '''
    Ejecuta el pipeline completo de feature engineering y exporta el resultado.

    Parametros (todos opcionales, usan las rutas del proyecto por defecto):
        data_clean_path     -- ruta a clean.csv desde preprocessing
        data_analysis_path  -- ruta a analysis.csv para metadatos
        vocab_unigrams_path -- ruta a rankings_unigrams.csv (opcional)
        analysis_json_path  -- ruta al analysis.json para entidades
        output_path         -- ruta de salida para features_nlp.csv

    Retorna el DataFrame con todas las features para uso inmediato.
    '''
    # Importaciones locales para evitar dependencias circulares al importar
    # este modulo desde main.py
    from feature_engineering.text_features    import compute_text_length_features
    from feature_engineering.keyword_features import load_vocabulary, compute_keyword_features
    from feature_engineering.pos_features     import load_spacy_model, compute_pos_features
    from feature_engineering.entity_features  import compute_entity_features

    logger.info('Iniciando pipeline de feature engineering')

    # Crear directorio de salida si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Carga del corpus desde clean.csv + metadatos de analysis.csv
    os.makedirs("data/unified",exist_ok=True)
    df_analysis=load_all_analysis_csv(DATA_DIR)
    df_analysis.to_csv("data/unified/analysis_unified.csv")
    df_clean=pd.read_csv(data_clean_path)
    df = df_clean.merge(df_analysis, on="indice", how="left")
    cleaned_series = df['comentario']
    corpus_list    = cleaned_series.tolist()
    json_file=load_all_analysis_json(DATA_DIR)
    with open("data/unified/analysis_json_unified.json","w") as f:
                json.dump(json_file,f)

    # Carga del modelo spaCy (se comparte entre pos y entity features)
    nlp = load_spacy_model()

    # Feature grupo 1: longitud y composicion textual
    logger.info('Calculando features de longitud de texto')
    text_feats = compute_text_length_features(cleaned_series)

    # Feature grupo 2: frecuencia y densidad de keywords del vocabulario
    logger.info('Calculando features de keywords')
    vocab_path = Path(vocab_unigrams_path)
    if vocab_path.exists():
        vocabulary  = load_vocabulary(vocab_unigrams_path)
        kw_feats    = compute_keyword_features(
            corpus        = corpus_list,
            token_counts  = text_feats['token_count'].to_numpy(),
            vocabulary    = vocabulary,
            ngram_n       = 1,
            index         = cleaned_series.index,
        )
    else:
        logger.warning('Archivo de vocabulario no encontrado: %s. Saltando keyword features.', vocab_unigrams_path)
        kw_feats = pd.DataFrame(index=cleaned_series.index)

    # Feature grupo 3: distribucion de partes del discurso
    logger.info('Calculando features de distribucion POS')
    pos_feats = compute_pos_features(cleaned_series, nlp)

    # Feature grupo 4: densidad y conteo de entidades nombradas
    logger.info('Calculando features de entidades nombradas')
    entity_feats = compute_entity_features(
        cleaned_series    = cleaned_series,
        nlp               = nlp,
        analysis_json_path = "data/unified/analysis_json_unified.json",
    )

    # Construccion del DataFrame final
    # Se preservan los metadatos demograficos del documento original
    demographic_cols = [col for col in DEMOGRAPHIC_COLUMNS if col in df.columns]
    metadata_cols = ['indice'] + demographic_cols if 'indice' in df.columns else demographic_cols
    metadata_df = df[metadata_cols].copy() if metadata_cols else df[['indice']].copy() if 'indice' in df.columns else pd.DataFrame(index=df.index)

    features_df = pd.concat(
        [metadata_df, text_feats, kw_feats, pos_feats, entity_feats],
        axis=1,
    )

    # Exportar resultado
    features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(
        'Features exportadas: %d documentos x %d columnas -> %s',
        len(features_df),
        len(features_df.columns),
        output_path,
    )

    return features_df


if __name__ == '__main__':
    result = run_feature_pipeline()
    logger.info('Pipeline finalizado. Shape: %s', result.shape)
    logger.info('Columnas: %s', list(result.columns))
