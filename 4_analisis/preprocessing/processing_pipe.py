import pandas as pd
import json
from .individual_functions import *

def process_pipeline(input_path):
    df = pd.read_csv(input_path, encoding="utf-8")
    df = normalize_columns(df)

    results_stage_1_2 = []
    results_stage_3 = []
    json_output = []

    for idx, row in df.iterrows():
        comentario = row["comentarios"]
        estrellas = row.get("cantidad_de_estrellas", None)

        text = normalize_text(comentario)
        text = remove_noise(text)

        pos_tags, noun_phrases, entities, entity_density = linguistic_analysis(text)

        clean_text = heavy_processing(text)

        # CSV 1 
        results_stage_1_2.append({
            "indice": idx,
            "estrellas": estrellas,
            "comentario": text,
            "pos_tags": pos_tags,
            "noun_phrases": noun_phrases,
            "entities": entities,
            "entity_density": entity_density
        })

        # CSV 2 
        results_stage_3.append({
            "indice": idx,
            "comentario_clean": clean_text
        })

        # JSON
        json_output.append({
            "indice": idx,
            "comentario": text,
            "pos_tags": pos_tags,
            "noun_phrases": noun_phrases,
            "entities": entities,
            "entity_density": entity_density
        })

    return results_stage_1_2, results_stage_3, json_output