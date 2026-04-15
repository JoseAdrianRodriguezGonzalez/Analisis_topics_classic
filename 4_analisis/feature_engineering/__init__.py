'''
feature_engineering/
--------------------
Bloque 3 del pipeline NLP: calculo de features numericas por documento.

Modulos:
    text_features    -- longitud de texto (tokens, caracteres, promedio)
    keyword_features -- frecuencia y densidad de keywords del vocabulario
    pos_features     -- distribucion de partes del discurso (POS tags)
    entity_features  -- densidad y conteo de entidades nombradas (NER)
    features         -- orquestador principal, punto de entrada del bloque
'''

from feature_engineering.features import run_feature_pipeline

__all__ = ['run_feature_pipeline']
