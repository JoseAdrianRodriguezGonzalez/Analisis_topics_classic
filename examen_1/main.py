import pandas as pd 
from utils.exploration import find_files 
from src.main import vectorization
import os 
emotion_values={
    "happy":(3.5,0.7),
    "surprise":(0.4,3.8),
    "neutral":(1.1,-1.3),
    "sad":(-3.4,-1.5),
    "fear":(-1.8,3),
    "disgust":(-2.6,1.9),
    "angry":(-3.2,2.3)
}
def create_dataframe(bebidas,out_path,buscador):
    bebidas_sorted=sorted(bebidas)
    files_drink =buscador.get_files(bebidas_sorted)
    rows=[]
    for directories_drink,drink in zip(files_drink,bebidas):
        files=[file for file in sorted(directories_drink) if os.path.splitext(file)[0].isdigit()]
        complete_paths=[os.path.join(buscador.root,drink,file) for file in files]
        for path,user_id in zip(complete_paths,files):
            vect=vectorization(path,emotion_values,3)
            vect.split_time()
            vect.histogram()
            vectors=vect.vectorize()
            for i,(valence,arousal) in vectors.items():
                rows.append({"drink":drink,
                             "user_id":os.path.splitext(user_id)[0],
                             "stage":i,
                             "valence":valence,
                             "arousal":arousal})
    df=pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df 
def main():
    buscador=find_files("data/bebidas")
    create_dataframe(["Pulque","Sake","Sidra","Tepache"],"data/processed/vectors.csv",buscador)
#    vect=vectorization(os.path.join("data/bebidas/Pulque",files[0][0]),emotion_values,3)
#    mascaras=vect.split_time()
#    vect.histogram()
#    vectores =vect.vectorize()
    
if __name__ =="main":
    main()
