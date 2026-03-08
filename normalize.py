import pandas as pd
import unicodedata
import re
import math
import string
import spacy
from collections import Counter
from nltk.util import ngrams

# IMPORTANT: Install spanish model for spacy
# python -m spacy download es_core_news_sm

# ======================================== #
#              NORMALIZATION               #
# ======================================== #
def normalize_text(text):
    if(pd.isna(text)):
        return text
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text

def clean_ages(value):
    if(pd.isna(value)):
        return None
    match = re.search(r"\d+\.?\d*", str(value))
    if(match):
        number = float(match.group())
        return math.floor(number)
    return None

def remove_line_breaks(text):
    if(pd.isna(text)):
        return text
    return str(text).replace("\n", " ").replace("\r", " ").strip()

# ======================================== #
#            FIX PLACE OF ORIGIN           #
# ======================================== #
municipios_guanajuato = [
    "abasolo","acambaro","apaseo el alto","apaseo el grande","atarjea",
    "celaya","comonfort","coroneo","cortazar","cueramaro",
    "doctor mora","dolores hidalgo","guanajuato","huanimaro","irapuato",
    "jaral del progreso","jerécuaro","leon","manuel doblado","moroleon",
    "ocampo","penjamo","pueblo nuevo","purisima del rincon",
    "romita","salamanca","salvatierra","san diego de la union",
    "san felipe","san francisco del rincon","san jose iturbide",
    "san luis de la paz","santa catarina","santiago maravatío",
    "silao","tarandacuao","tarimoro","tierra blanca",
    "uriangato","valle de santiago","victoria","villagran",
    "xichu","yuriria"
]

municipios_estados=["orizaba","michoacan","oaxaca","arandas"]

def fix_place_of_origin(value):
    if(pd.isna(value)):
        return "otro"
    text = str(value)
    municipios_mix = municipios_guanajuato + municipios_estados
    for municipio in municipios_mix:
        if(municipio in text):
            return municipio
    return "otro"

# ======================================== #
#               NLP FUNCTIONS              #
# ======================================== #
def remove_punct_and_lower_global(text):
    if(pd.isna(text)):
        return text
    text = str(text).lower()
    spanish_signs = string.punctuation + "¿¡"
    text =  text.translate(str.maketrans("", "", spanish_signs))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_nlp_tokens(text, nlp_model):
    if(pd.isna(text) or text == ""):
        return []

    doc = nlp_model(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip() != "":
            tokens.append(token.lemma_)
    return tokens

def build_ngrams_and_frequency(texts_tokens, n):
    all_ngrams = []
    for tokens in texts_tokens:
        if(len(tokens) >= n):
            n_grams = list(ngrams(tokens, n))
            n_grams_str = [" ".join(gram) for gram in n_grams]
            all_ngrams.extend(n_grams_str)

    frequency_count = Counter(all_ngrams)
    total_ngrams = sum(frequency_count.values())

    results = []
    for ngram, count in frequency_count.most_common():
        relative_frequency = count / total_ngrams if total_ngrams > 0 else 0
        results.append((ngram, count, relative_frequency))
    
    df_frequencies = pd.DataFrame(results, columns=["ngram", "total_frequency", "relative_frequency"])
    return df_frequencies

def pipeline_normalize_raw_data():
    print("---"*18)
    print("# Conjunto de datos del primer instrumento #")
    print("---"*18)
    df = pd.read_excel("data/raw/data_956.xlsx")
    df.columns = (
        df.columns.str.strip().str.lower().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    print("Step 1 completed: Normalize columns")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(normalize_text)
    print("Step 2 completed: Normalization")
    df = df.drop_duplicates(subset=["correo electronico"])
    print("Step 3 completed: Delete duplicate emails")
    df["cual es tu edad?"] = df["cual es tu edad?"].apply(clean_ages) 
    print("Step 4 completed: Clean ages")
    df["de que lugar procedes?"] = df["de que lugar procedes?"].apply(fix_place_of_origin)
    print("Step 5 completed: Fix place of origin")
    df["en el siguiente espacio, expresa tu opinion general acerca de salamanca"] = df["en el siguiente espacio, expresa tu opinion general acerca de salamanca"].apply(remove_line_breaks)
    print("Step 6 completed: Remove line breaks")
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Step 7 completed: Save cleaned data")
    print("---"*21)
    print("# Conjunto de datos del segundo instrumento instrumento #")
    print("---"*21)
    df = pd.read_csv("data/raw/data_survey.csv")
    df.columns = (
        df.columns.str.strip().str.lower().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    print("Step 1 completed: Normalize columns")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(normalize_text)
    print("Step 2 completed: Normalization")
    #df = df.drop_duplicates(subset=["correo electronico"])
    #print("Step 3 completed: Delete duplicate emails")
    df["edad"] = df["edad"].apply(clean_ages) 
    print("Step 3 completed: Clean ages")
    df["ciudad de origen"] = df["ciudad de origen"].apply(fix_place_of_origin)
    print("step 4 completed: fix place of origin")
    print(df.columns)
    df["en tus propias palabras, que opinas de salamanca y que palabra, frase o recuerdo usarias para describirla principalmente?"] = df["en tus propias palabras, que opinas de salamanca y que palabra, frase o recuerdo usarias para describirla principalmente?"].apply(remove_line_breaks)
    print("Step 5 completed: Remove line breaks")
    df.to_csv("data/processed/cleaned_data_survey_2.csv", index=False)
    print("Step 6 completed: Save cleaned data")
    print("---"*21)
    print("# Uniendo conjunto de datos y normalizando columnas #")
    print("---"*21)
    df_final=join_datasets("data/processed/cleaned_data.csv","data/processed/cleaned_data_survey_2.csv")
    df_final.to_csv("data/processed/data_basis.csv")
