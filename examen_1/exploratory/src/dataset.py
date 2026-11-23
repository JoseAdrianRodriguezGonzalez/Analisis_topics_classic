import pandas as pd
import os
def read_csv(path:str)->dict[str,pd.DataFrame]:
    diccionario=dict()
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path,folder)):
            if "Resumen.csv" in file:
                path_=os.path.join(path,folder,file)
                df=pd.read_csv(path_)
                diccionario[folder]=df
    return diccionario

"""
def  transpose_matrix(diccionario_df:dict[str,pd.DataFrame],emociones:list[str],start:str,end:str):
    df_all=pd.DataFrame()
    for liquor, df in diccionario_df.items():
       # print(df.columns)
        df_temp=df.loc[:,3:26].copy()
        df_temp.index=emociones
        df_temp=df_temp.to_frame().T 
        df_temp["liquor"]=liquor
        df_all=pd.concat([df_all,df_temp],ignore_index=True)
    return df_all 
"""
def extract_embedding(path):
    df=pd.read_csv(path,sep="\t",encoding="utf-8")
    emotions = ["anger", "fear", "sadness", "disgust", "happiness", "surprise"]
    df1 = df[emotions]
    return df1
def transpose(df:pd.pandas):
    usuarios=[ col for col in df.columns if col.isdigit()]
    emociones=df.loc[df["Unnamed: 1"].notna(),["Unnamed: 1"]+usuarios]
    dataframe=emociones.set_index("Unnamed: 1")

    dataframe=dataframe.T 
    return dataframe
