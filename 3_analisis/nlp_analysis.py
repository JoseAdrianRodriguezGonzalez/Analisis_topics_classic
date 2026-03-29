import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re
import math
import string
import spacy
import os
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud

# IMPORTANT: Install spanish model for spacy
# python -m spacy download es_core_news_sm

# ======================================== #
#          GENERAL NORMALIZATION           #
# ======================================== #
def clean_general_text(text):
    if(pd.isna(text)):
        return text

    text = str(text).strip().lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    spanish_signs = string.punctuation + "¿¡"
    text = text.translate(str.maketrans("","", spanish_signs))
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_text_light(text):
    if(pd.isna(text)):
        return text
    
    text = str(text).strip().lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def remove_accents_and_punct(text):
    if(pd.isna(text)):
        return text
    
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    spanish_signs = string.punctuation + "¿¡"
    text = text.translate(str.maketrans("", "", spanish_signs))
    
    return text

# ======================================== #
#          SPECIFIC NORMALIZATION          #
# ======================================== #
def clean_ages(value):
    if(pd.isna(value)):
        return None
    match = re.search(r"\d+\.?\d*", str(value))
    if(match):
        number = float(match.group())
        return math.floor(number)
    return None

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

STOPWORDS_EXTRA = {"el", "la", "lo", "le", "yo", "tu", "mi", "su", "un", "al", "etc", "nan"}

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
def process_nlp_tokens(text, nlp_model):
    if(pd.isna(text) or text == ""):
        return []

    protected_words = set(municipios_guanajuato + municipios_estados)

    doc = nlp_model(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip() != "":
            if(len(token.text.strip()) <= 2):
                continue
            if re.sub(r'[^a-zA-Z0-9]', '', token.text.strip()).isdigit():
                continue
            if(token.text.lower() in {"etc", "asi", "tmb", "tqm"}):
                continue

            # 1. Remove accents to the original token to check if it is a protected word
            word_no_accents = remove_accents_and_punct(token.text)
            if(word_no_accents in protected_words):
                tokens.append(word_no_accents)
            else:
                # 2. If it is not a protected word, lemmatize it, then remove accents
                lemma_no_accents = remove_accents_and_punct(token.lemma_)

                # Filtrar si el lemma contiene alguna palabra de <= 2 chars o en STOPWORDS_EXTRA
                partes = lemma_no_accents.split()
                partes_limpias = [p for p in partes if len(p) > 2 and p not in STOPWORDS_EXTRA]
                if not partes_limpias:
                    continue
                # Usar solo las partes limpias
                token_limpio = " ".join(partes_limpias)
                tokens.append(token_limpio)
                #print(f"APPEND: text={token.text!r} lemma={lemma_no_accents!r}")
                #tokens.append(lemma_no_accents)
    return tokens

def build_ngrams_and_frequency(texts_tokens, n):
    all_ngrams = []
    for tokens in texts_tokens:
        #verificar que cada token tenga una longitud de al menos 3 caracteres
        temporal = [token for token in tokens if len(token) <= 3]
        #print(f"Tokens removed for being too short: {temporal}")
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

# ======================================== #
#              VISUALIZATION               #
# ======================================== #
def generate_wordcloud(text_series, output_path="data/processed/wordcloud.png"):
    all_text = " ".join(text_series.dropna().astype(str))

    if not all_text.strip():
        print("WARNING: No text to generate wordcloud")
        return

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        colormap="viridis",
        max_words=100
    ).generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
# ======================================== #
#                  PIPELINE                #
# ======================================== #
def pipeline_nlp_analysis(input_csv="data/processed/data_basis.csv", output_csv="data/processed/data_nlp.csv"):
    print("---"*10)
    print("STARTING NLP PIPELINE")
    print("---"*10)

    #           LOAD DATA             #
    df = pd.read_csv(input_csv)
    if("Unnamed: 0" in df.columns):
        df = df.drop(columns=["Unnamed: 0"])
    print("Step 1: Data loaded")

    #         CLEAN DATA              #
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if(col == "comentario"):
            df[col] = df[col].apply(clean_text_light)
        else:
            df[col] = df[col].apply(clean_general_text)
    print("Step 2: Data cleaned")

    #      CLEAN SPECIFIC DATA        #
    if("lugar" in df.columns):
        df["lugar"] = df["lugar"].apply(fix_place_of_origin)
    if("edad" in df.columns):
        df["edad"] = df["edad"].apply(clean_ages)
    print("Step 3: Specific data cleaned")

    #         LOAD NLP MODEL          #
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        print("Downloading spacy model...")
        os.system("python -m spacy download es_core_news_sm")
        nlp = spacy.load("es_core_news_sm")
    print("Step 3: NLP model loaded")

    #   TOKENIZE-STOPWORDS-LEMMATIZE  #
    #  This process only applies to the "comentario" column
    if("comentario" in df.columns):
        df["comentario_tokens"] = df["comentario"].apply(lambda x: process_nlp_tokens(x, nlp))
        df["comentario_cleaned"] = df["comentario_tokens"].apply(lambda x: " ".join(x))
        print("Step 4: Tokenize-stopwords-lemmatize done")
    else: 
        print("WARNING: Column 'comentario' not found. Skipping NLP")
        return

    #           BUILD NGRAMS          #
    tokens_list = df["comentario_tokens"].tolist()
    #Exportar los ngramas a csv
    df_unigrams = build_ngrams_and_frequency(tokens_list, 1)
    df_bigrams = build_ngrams_and_frequency(tokens_list, 2)
    df_trigrams = build_ngrams_and_frequency(tokens_list, 3)
    print("Step 5: Ngrams built")

    #           SAVE DATA             #
    df.drop(columns=["comentario_tokens"]).to_csv(output_csv, index=False)

    rankings_path = "data/processed/rankings_frequencies.xlsx"
    with pd.ExcelWriter(rankings_path) as writer:
        df_unigrams.to_excel(writer, sheet_name="unigrams", index=False)
        df_bigrams.to_excel(writer, sheet_name="bigrams", index=False)
        df_trigrams.to_excel(writer, sheet_name="trigrams", index=False)
    print("Step 6: Rankings saved")

    #       GENERATE WORDCLOUDS       #
    generate_wordcloud(df["comentario_cleaned"])
    print("Step 7: Wordcloud generated")   

    print("---"*10)
    print("NLP PIPELINE FINISHED")
    print("---"*10)

if(__name__ == "__main__"):
    pipeline_nlp_analysis()