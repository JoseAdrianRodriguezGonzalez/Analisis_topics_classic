import re
import spacy
import pandas as pd
from unidecode import unidecode

nlp = spacy.load("es_core_news_sm")

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

def tokenize(text):
    document = nlp(text)
    return [token.text for token in document]


# =================================== #
#         Linguistic Analysis         #
# =================================== #
def linguistic_analysis(text):
    document = nlp(text)

    pos_tags = [token.pos_ for  token in document]

    noun_phrases = [chunk.text for chunk in document.noun_chunks]

    entities = [{"text": ent.text, "label": ent.label_} for ent in document.ents]

    entity_density = len(document.ents) / len(document) if len(document) > 0 else 0

    return pos_tags, noun_phrases, entities, entity_density

# =================================== #
#           Heavy processing          #
# =================================== #
def heavy_processing(text):
    document = nlp(text.lower())

    tokens = []
    for token in document:
        if not token.is_stop and not token.is_punct:
            clean = unidecode(token.text)
            tokens.append(clean)

    return " ".join(tokens) 