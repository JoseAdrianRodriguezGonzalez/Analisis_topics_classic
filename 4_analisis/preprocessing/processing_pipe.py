import pandas as pd
import json
from .individual_functions import *

def process_pipeline(input_path):
    df = pd.read_csv(input_path, encoding="utf-8")
    df = normalize_columns(df)

    spanish_results = []
    english_results = []
    mixed_results = []

    for idx, row in df.iterrows():
        comentario = row["comentarios"]
        estrellas = row.get("cantidad_de_estrellas", None)

        text = normalize_text(comentario)
        text=remove_light_noise(text)
        # Detection
        lang = detect_language_type(text)
        nlp = get_nlp_model(lang)

        if nlp is None:
            mixed_results.append({"indice": idx, "comentario": text})
            continue

        pos_tags, noun_phrases, entities, entity_density = linguistic_analysis(text, nlp)
        text = remove_noise(text)
        clean_text = heavy_processing(text, nlp)

        # CSV 1 
        results_1_2 = {
            "indice": idx,
            "estrellas": estrellas,
            "comentario": text,
            "pos_tags": pos_tags,
            "noun_phrases": noun_phrases,
            "entities": entities,
            "entity_density": entity_density
        }

        # CSV 2 
        results_3 = {
            "indice": idx,
            "comentario_clean": clean_text
        }
    
        if lang == "es":
            spanish_results.append((results_1_2, results_3))
        elif lang == "en":
            english_results.append((results_1_2, results_3))
        else:
            mixed_results.append({"indice": idx, "comentario": text})

    return spanish_results, english_results, mixed_results
