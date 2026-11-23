import os 
import pandas as pd 
import numpy as np 
class vectorization:
    def __init__(self,path,emotion_values,stages) -> None:
        self.root=path 
        self.list_emotion=sorted([emotion for emotion in emotion_values])
        self.emotion_values=emotion_values
        self.emotion_matrix=np.array(
            [self.emotion_values[e] for e in self.list_emotion],
            dtype=np.float32
        )
        self.num_stages=stages
        self.df=pd.read_csv(path)
    def split_time(self):
        print(self.df["time_sec"].dtype)
        time_column=self.df["time_sec"]
        tf=time_column.iloc[-1]
        t0=time_column.iloc[0]
        duration=tf-t0
        delta=duration/self.num_stages 
        fases={}
        for i in range(self.num_stages):
            t_start=t0+i*delta 
            t_end=t0+(i+1)*delta
            if i ==self.num_stages-1:
                mask=(time_column>=t_start) & (time_column<=t_end)
            else:
                mask = (time_column>=t_start) &(time_column <t_end)
            fases[f"Fases{i+1}"]=self.df[mask]
        self.stages=fases
        return fases
    def histogram(self):
        histograms={}
        for stage,df_stage in self.stages.items():
            counts=(
                df_stage["emotion"].value_counts()
                .reindex(self.list_emotion,fill_value=0).to_numpy(dtype=np.float32)
            )
            histograms[stage]=counts 
        self.histograms=histograms
        return histograms
    def vectorize(self):
        vectors= {}
        for stage,counts in self.histograms.items():
            c_norm=counts/counts.sum()
            v=c_norm@self.emotion_matrix
            vectors[stage]=v
        self.vectors=v
        return vectors 

class find_files:
    def __init__(self,root:str) -> None:
        self.root=root 
    def find_file(self,path,directory=None):
        main_direct=os.path.join(self.root,directory) if directory != None else self.root 
        if path in os.listdir(main_direct):
            print("the file is here")
            df=pd.read_csv(os.path.join(main_direct,path)) 
            print(df)
        else:
            print("This file is not here")
            return 
    def show_files(self,path):
        for files in os.listdir(os.path.join(self.root,path)):
            print(files)
    def get_files(self,path):
        if isinstance(path,str):
            path_finder=os.path.join(self.root,path)
            return sorted(os.listdir(path_finder))
        list_files=[]
        for directories in path:
            file=sorted(os.listdir(os.path.join(self.root,directories)))
            list_files.append(file)
        return list_files
                


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
    print("si")
    buscador=find_files("../data/bebidas")
    create_dataframe(["Pulque","Sake","Sidra","Tepache"],"../data/processed/vectors.csv",buscador)

    
if __name__ =="main":
    main()
