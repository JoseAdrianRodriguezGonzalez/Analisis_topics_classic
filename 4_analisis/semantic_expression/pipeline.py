import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from .ner import *
from .vectorization import *
from .BERTopic import *
import joblib
from tqdm import tqdm
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
def extract_group_ner():
    paths=[
        "data/data_spanish/analysis.json",
        "data/data_english/analysis.json",
        "data/data_mixed/analysis.json"
    ]
    all_data=[]
    for path in paths:
        data=read_json(path)
        all_data.extend(data)
    cleaned=clean_entities(all_data)
    aggregate=aggregate_entities(cleaned)
    merged=merge_similar_entities(aggregate)
    noun=enrichment_text(merged,all_data)
    return noun 
def build_doc_entity_map(analysis_paths):
    doc_entities = {}
    for path in analysis_paths:
        data =read_json(path)
        for row in data:
            idx = row["indice"]
            ents = [
                normalize_ner(e["text"])
                for e in row["entities"]
                if len(e["text"]) >= 3
            ]
            if idx not in doc_entities:
                doc_entities[idx] = []
            doc_entities[idx].extend(ents)
    return doc_entities
def enrich_texts_with_ner(df, doc_entities):
    enriched_texts = []
    for _, row in df.iterrows():
        idx = row["indice"]
        text = row["comentario_clean"]
        ents = doc_entities.get(idx, [])
        ent_tokens = " ".join([f"__ent_{e}__" for e in ents])
        enriched_texts.append(text + " " + ent_tokens)
    return enriched_texts
def pipe():
    create_folder("data/features")
    create_folder("data/models")
    create_folder("data/results")
    analysis_ner=extract_group_ner()
    df=pd.read_csv("data/translations/normalized_spanish.csv")
    df["comentario_clean"]=df["comentario_clean"].fillna("").astype(str)
    doc_entities=build_doc_entity_map([
        "data/data_spanish/analysis.json",
        "data/data_english/analysis.json",
        "data/data_mixed/analysis.json"
    ])
    texts=enrich_texts_with_ner(df,doc_entities)
    features=build_features(texts)
    with open("data/features/ner_groups.json","w") as f:
        json.dump(analysis_ner,f,indent=4)
    top_words=get_top_tfidf_words(features["X_tfidf"],features["vectorizer"])
    with open("data/features/entities_top_words.json","w") as f:
        json.dump(top_words,f, indent=4)
    print(features["X_tfidf"])
    print(features["X_yake"])
    joblib.dump(features["vectorizer"], "data/models/tfidf.pkl")
    joblib.dump(features["vectorizer_yake"], "data/models/yake_vectorizer.pkl")
    topicBERT=BERTopic_analysis(None,None,None,texts)
    embedding=topicBERT.embedding_extraction(None,None)
    np.save("data/features/docs_with_topics.npy",embedding)
    topics,probs=topicBERT.fit()
    topic_info=topicBERT.get_topics()
    df["topic"] = topics
    df.to_csv("data/results/docs_with_topics.csv", index=False)
    topic_info.to_csv("data/results/topics.csv", index=False)
    topicBERT.model.save("data/models/bertopic_model")
    print(topic_info.head())
def pipe_microtopics():
    df=pd.read_csv("data/results/docs_with_topics.csv")
    df["comentario_clean"]=df["comentario_clean"].fillna("").astype(str)
    micro_results=[]
    doc_entities=build_doc_entity_map([
        "data/data_spanish/analysis.json",
            "data/data_english/analysis.json",
            "data/data_mixed/analysis.json"
    ])
    for region in tqdm(df["location"].unique(),desc="Regions"):
        for topic in tqdm(df["topic"].unique(), desc=f"Topics {region}",leave=False):
            subset=df[
                (df["location"]==region)&(df["topic"]==topic)
            ]
            if len(subset) <30:
                continue
            texts=enrich_texts_with_ner(subset,doc_entities)
            model=BERTopic_analysis(None,None,None,texts)
            topics_micro,_ =model.fit()
            subset=subset.copy()
            subset["microtopic"]=topics_micro
            subset["parent_topic"]=topic
            subset["region"]=region
            micro_results.append(subset)
    if len(micro_results)>0:
        df_micro=pd.concat(micro_results,ignore_index=True)
        df_micro.to_csv("data/results/microtopics.csv",index=False)


