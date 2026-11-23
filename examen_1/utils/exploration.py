import os
from uuid import main
from numpy import sort
import pandas as pd 
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
                
