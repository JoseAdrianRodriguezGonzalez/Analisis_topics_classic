import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
class BaseAnalysis_:
    def __init__(self,df:pd.DataFrame):
        self.df=df
import re


def clean_filename(name):
    name = name.lower()
    name = re.sub(r"[^\w\s/-]", "", name)   # ahora permite /
    name = name.replace(" ", "_")
    return name
class DA(BaseAnalysis_):
    def info_column(self,column_name:str):
        col=self.df[column_name]
        descriptive={
            "mean":col.mean(),
            "median":col.median(),
            "std":col.std(),
            "min":col.min(),
            "max":col.max(),
            "q1":col.quantile(0.25),
            "q3":col.quantile(0.75),
            "skewness":col.skew(),
            "kurtosis":col.kurtosis()
        }
        return descriptive
    def plot_histogram(self,column_name:str,bins:int=5,save_path=None):
        col=self.df[column_name]
        plt.figure()
        plt.hist(col,bins=bins)
        plt.title(f"Histograma de {column_name}")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.xticks([1,2,3,4,5])
        plt.grid(axis="y",linestyle="--",alpha=0.7)
        if save_path is not None:
            plt.savefig(clean_filename(save_path)+".png",dpi=300,bbox_inches="tight")
        else:
            plt.show()
    def plot_boxplots(self,column_name:str,save_path=None):
        col=self.df[column_name]
        plt.figure()
        plt.boxplot(col,vert=True)
        plt.title(f"Boxplot de {column_name}")
        plt.xlabel("Valor")
        plt.xticks([1], [column_name])
        if save_path is not None:
            plt.savefig(clean_filename(save_path)+".png",dpi=300,bbox_inches="tight")
        else:
            plt.show()


