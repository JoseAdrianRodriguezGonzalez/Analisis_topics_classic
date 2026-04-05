import os
import re
import spacy
import json
import pandas as pd
from unidecode import unidecode
from langdetect import detect_langs, LangDetectException, DetectorFactory

DetectorFactory.seed = 0

nlp_es = spacy.load("es_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# =================================== #
#             Normalize CSV           #
# =================================== #
def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

# =================================== #
#            Lang detection           #
# =================================== #
def detect_language_type(text):
    try:
        langs = detect_langs(text)

        if not langs:
            return "unknown"
        
        probs = {l.lang: l.prob for l in langs}

        if "es" in probs and "en" in probs:
            if probs["es"] > 0.05 and probs["en"] > 0.05:
                return "mixed"
        
        if probs.get("es", 0.0) > 0.85:
            return "es"
        elif probs.get("en", 0.0) > 0.85:
            return "en"
        else:
            return "mixed"

    except LangDetectException:
        return "unknown"
    
def get_nlp_model(lang):
    if lang == "es":
        return nlp_es
    elif lang == "en":
        return nlp_en
    else:
        return None 

# =================================== #
#           Light processing          #
# =================================== #
def normalize_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_noise(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s.,]', '', text)
    return text
def remove_light_noise(text):
    text=re.sub(r'<*:?>','',text)
    return text
def tokenize(text, nlp):
    document = nlp(text)
    return [token.text for token in document]


# =================================== #
#         Linguistic Analysis         #
# =================================== #
def linguistic_analysis(text, nlp):
    document = nlp(text)
    pos_tags = [token.pos_ for  token in document]
    noun_phrases = [chunk.text for chunk in document.noun_chunks]
    entities = [{"text": ent.text, "label": ent.label_} for ent in document.ents]
    entity_density = len(document.ents) / len(document) if len(document) > 0 else 0
    return pos_tags, noun_phrases, entities, entity_density

# =================================== #
#           Heavy processing          #
# =================================== #
def heavy_processing(text, nlp):
    document = nlp(text.lower())

    tokens = []
    for token in document:
        if not token.is_stop and not token.is_punct:
            clean = unidecode(token.text)
            tokens.append(clean)

    return " ".join(tokens) 

# =================================== #
#                Results              #
# =================================== #
def create_data_folders():
    base = "data"
    paths = [
        f"{base}/data_spanish",
        f"{base}/data_english",
        f"{base}/data_mixed"
    ]

    for path in paths:
        os.makedirs(path, exist_ok = True)

def save_results(results, folder):
    if len(results) == 0:
        return 
    stage_1_2 = [r[0] for r in results]
    stage_3 = [r[1] for r in results]
    base_path = f"data/{folder}"
    os.makedirs(base_path, exist_ok = True)
    pd.DataFrame(stage_1_2).to_csv(f"{base_path}/analysis.csv", index=False)
    pd.DataFrame(stage_3).to_csv(f"{base_path}/clean.csv", index=False)
    with open(f"{base_path}/analysis.json", "w", encoding="utf-8") as f:
        json.dump(stage_1_2, f, indent=4, ensure_ascii=False)
    print(f"Results saved in {base_path}")
