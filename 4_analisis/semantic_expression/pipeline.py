import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from .ner import *
from .vectorization import *


def build_features(texts):
    X_tfidf,vectorizer = compute_tfidf(texts)
    yake_keywords=extract_yake(texts)
    yake_vocab=build_yake_vocab(yake_keywords)
    vectorizer_yake=CountVectorizer(
        vocabulary=yake_vocab,
        binary=True 
    )
    X_yake=vectorizer_yake.transform(texts) 
    return {
        "X_tfidf":X_tfidf,
        "vectorizer":vectorizer,
        "X_yake":X_yake,
        "yake_vocab":yake_vocab,
        "vectorizer_yake":vectorizer_yake
    }
def extract_group_ner(src):
    data=read_json(src)
    cleaned=clean_entities(data)
    aggregate=aggregate_entities(cleaned)
    merged=merge_similar_entities(aggregate)
    noun=enrichment_text(merged,data)
    return noun  
def pipe():
    analysis_ner=extract_group_ner("data/data_spanish/analysis.json")
    df=pd.read_csv("data/data_spanish/clean.csv")
    texts=df["comentario_clean"].astype(str).to_list()
    features=build_features(texts)
    top_words=get_top_tfidf_words(features["X_tfidf"],features["vectorizer"])
    create_folder("data/features")
    with open("data/features/entities.json","w") as f:
        json.dump(top_words,f, indent=4)
    print(features["X_tfidf"])
    print(features["X_yake"])
