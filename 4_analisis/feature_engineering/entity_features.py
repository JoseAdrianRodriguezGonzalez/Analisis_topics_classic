'''
entity_features.py
------------------
Calcula features relacionadas con entidades nombradas (NER) por documento.

Puede operar en dos modos segun lo que este disponible:

    Modo A (preferido) -- lee el JSON de analisis ya generado por el pipeline
    de Adrian (4_analisis/data/data_spanish/analysis.json) que ya contiene
    las entidades extraidas y la densidad calculada por documento.

    Modo B (fallback) -- recalcula las entidades directamente desde el texto
    usando spaCy cuando el JSON de Adrian no esta disponible.

Tipos de entidades consideradas:
    GPE   -- entidades geopoliticas (paises, ciudades, estados)
    LOC   -- ubicaciones geograficas
    ORG   -- organizaciones
    FAC   -- instalaciones (edificios, aeropuertos, etc.)

Funciones:
    load_entities_from_json    -- carga entidades del JSON de Adrian (Modo A)
    compute_entities_from_text -- recalcula entidades con spaCy (Modo B)
    compute_entity_features    -- orquesta ambos modos y retorna DataFrame
'''

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENTITY_TYPES = ['GPE', 'LOC', 'ORG', 'FAC']


def load_entities_from_json(
    analysis_json_path: str | Path,
    n_documents: int,
) -> pd.DataFrame | None:
    '''
    Modo A: lee el archivo analysis.json generado por processing_pipe.py
    de Adrian y extrae entity_density y conteos por tipo.

    El JSON tiene la estructura:
        [{indice, entity_density, entities: [{text, label}], ...}, ...]

    Si el archivo no existe o tiene errores retorna None para que el
    llamador use el Modo B.

    Parametros:
        analysis_json_path -- ruta al analysis.json
        n_documents        -- total de documentos esperados (para alinear indices)
    '''
    path = Path(analysis_json_path)
    if not path.exists():
        logger.warning('JSON de entidades no encontrado en %s, usando Modo B', path)
        return None

    try:
        with open(path, encoding='utf-8') as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        logger.warning('Error al leer JSON de entidades: %s, usando Modo B', error)
        return None

    # Inicializar arrays vacios alineados con el total de documentos
    entity_density = np.zeros(n_documents, dtype=float)
    type_counts    = {etype: np.zeros(n_documents, dtype=int) for etype in ENTITY_TYPES}

    for record in data:
        idx = record.get('indice')
        if idx is None or idx >= n_documents:
            continue

        entity_density[idx] = record.get('entity_density', 0.0)

        for entity in record.get('entities', []):
            label = entity.get('label', '')
            if label in type_counts:
                type_counts[label][idx] += 1

    result = pd.DataFrame({'entity_density': np.round(entity_density, 6)})
    for etype in ENTITY_TYPES:
        result[f'entity_count_{etype.lower()}'] = type_counts[etype]

    result['entity_count_total'] = sum(type_counts[t] for t in ENTITY_TYPES)

    logger.info(
        'Entidades cargadas desde JSON: %d registros, densidad media=%.4f',
        len(data),
        result['entity_density'].mean(),
    )
    return result


def compute_entities_from_text(
    cleaned_series: pd.Series,
    nlp,
) -> pd.DataFrame:
    '''
    Modo B: recalcula entidades directamente desde el texto limpio usando spaCy.
    Se usa cuando el JSON de Adrian no esta disponible.

    La densidad se calcula como:
        entity_density = n_entidades / n_tokens_del_documento

    Esto es consistente con la formula usada en processing_pipe.py de Adrian.

    Parametros:
        cleaned_series -- serie de comentarios limpios
        nlp            -- modelo spaCy ya cargado
    '''
    rows = []
    for text in cleaned_series.fillna(''):
        if not text.strip():
            row = {'entity_density': 0.0}
            row.update({f'entity_count_{t.lower()}': 0 for t in ENTITY_TYPES})
            row['entity_count_total'] = 0
            rows.append(row)
            continue

        doc = nlp(text)
        n_tokens  = len(doc)
        n_ents    = len(doc.ents)
        density   = round(n_ents / n_tokens, 6) if n_tokens > 0 else 0.0

        counts = {etype: 0 for etype in ENTITY_TYPES}
        for ent in doc.ents:
            if ent.label_ in counts:
                counts[ent.label_] += 1

        row = {'entity_density': density}
        row.update({f'entity_count_{t.lower()}': counts[t] for t in ENTITY_TYPES})
        row['entity_count_total'] = n_ents
        rows.append(row)

    result = pd.DataFrame(rows, index=cleaned_series.index)

    logger.info(
        'Entidades recalculadas con spaCy: %d documentos, densidad media=%.4f',
        len(result),
        result['entity_density'].mean(),
    )
    return result


def compute_entity_features(
    cleaned_series: pd.Series,
    nlp,
    analysis_json_path: str | Path | None = None,
) -> pd.DataFrame:
    '''
    Orquesta Modo A y Modo B segun disponibilidad del JSON de Adrian.

    Intenta primero Modo A (leer JSON). Si no esta disponible o falla,
    cae al Modo B (recalcular con spaCy).

    Columnas de salida:
        entity_density        -- entidades / tokens del documento
        entity_count_gpe      -- conteo de entidades tipo GPE
        entity_count_loc      -- conteo de entidades tipo LOC
        entity_count_org      -- conteo de entidades tipo ORG
        entity_count_fac      -- conteo de entidades tipo FAC
        entity_count_total    -- suma de todas las entidades relevantes
    '''
    if analysis_json_path is not None:
        result = load_entities_from_json(analysis_json_path, n_documents=len(cleaned_series))
        if result is not None:
            result.index = cleaned_series.index
            return result

    logger.info('Ejecutando Modo B: recalculo de entidades con spaCy')
    return compute_entities_from_text(cleaned_series, nlp)
