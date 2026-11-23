import numpy as np 
import pandas as pd
from pandas.core.dtypes.cast import NumpyArrayT 
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
        

