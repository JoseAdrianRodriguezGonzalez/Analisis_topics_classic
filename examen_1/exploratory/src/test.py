import pandas as pd  
from transforms import normalize_counts
 
embedding=pd.read_csv("../data/raw/AffectVec-data.tsv",sep="\t",encoding="utf-8")
df=pd.read_csv("../out/new_vision.csv")
emotions = ["anger", "fear", "sadness", "disgust", "happiness", "surprise"]
print(df)
print(embedding)
embedding = embedding[emotions]


