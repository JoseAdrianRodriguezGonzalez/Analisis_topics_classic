import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from .ner import *
from .vectorization import *
from .BERTopic import *
import joblib
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
    create_folder("data/features")
    create_folder("data/models")
    create_folder("data/results")
    analysis_ner=extract_group_ner("data/data_spanish/analysis.json")
    df=pd.read_csv("data/translations/normalized_spanish.csv")
    texts=df["comentario_clean"].astype(str).to_list()
    features=build_features(texts)
    top_words=get_top_tfidf_words(features["X_tfidf"],features["vectorizer"])
    with open("data/features/entities.json","w") as f:
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
