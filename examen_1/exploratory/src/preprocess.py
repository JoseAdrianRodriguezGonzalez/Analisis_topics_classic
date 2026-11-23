import os
import pandas as pd
from dataset import read_csv ,transpose
emotions = ["anger", "fear", "sadness", "disgust","neutral", "happiness", "surprise"]
diccionario=read_csv("../data/bebidas/")
#Casos puntuales por bebdia 
df_pulque=diccionario["Pulque"]
df_sake=diccionario["Sake"]
df_sidra=diccionario["Sidra"]
df_tepache=diccionario["Tepache"]
df_complete=pd.DataFrame()
dfs=[]
for keys,values in diccionario.items():
    df_temp=transpose(values)
    df_temp["liquor"]=keys
    dfs.append(df_temp)
df_complete = pd.concat(dfs,axis=0,ignore_index=False)
df_complete.to_csv("../out/new_vision.csv")
print(df_complete)
