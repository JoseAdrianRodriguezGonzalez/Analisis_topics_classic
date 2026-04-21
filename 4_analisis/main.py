import pandas as pd
#primer paso 
from preprocessing.individual_functions import *
from preprocessing.processing_pipe import process_pipeline
# paso 2 
from feature_engineering.vocabulary import build_vocabulary_from_clean
from feature_engineering.features import run_feature_pipeline
from translation.pipeline import normalize_language 
from semantic_expression.vectorization import create_folder
# paso 3 y 4  
from semantic_expression.pipeline import pipe
# paso 5 
from clustering.clustering_pipeline import run_clustering_pipeline

main_csv="data/raw/complete.csv"
def preprocessing_complete(input_file):
    create_data_folders()
    spanish, english, mixed = process_pipeline(input_file)
    save_results(spanish, "data_spanish")
    save_results(english, "data_english")
    save_results(mixed,"data_mixed")
    #if len(mixed) > 0:
    #    pd.DataFrame(mixed).to_csv("data/data_mixed/mixed.csv", index=False)
    #    print("⚠️  File generated for mixed results.")
    print("\nPipeline executed successfully.")
    print("Unir idiomas")
    create_folder("data/translations/")
    df_spanish=pd.read_csv("data/data_spanish/clean.csv")
    df_english=pd.read_csv("data/data_english/clean.csv")
    df_mixed=pd.read_csv("data/data_mixed/clean.csv")
    df_joined=pd.concat([df_spanish,df_english,df_mixed],ignore_index=True)
    df_joined.to_csv("data/translations/joined.csv",index=False)
    print("Dataframe con el idioma junto creado")
def main():
    #df=create_csv_master("data/raw",main_csv)  
    print("Primera fase: preprocesamiento")
    #preprocessing_complete(main_csv)
    print("Traducir para normalizar spanish")
    df_joined=pd.read_csv("data/translations/joined.csv")
    df=normalize_language(df_joined)
    df.to_csv("data/translations/normalized_spanish.csv",index=False)
    print("Se normalizo para los embeddings")
    print("Segunda fase: embeddings y topicos ")
    pipe()
    #print("Tercera fase: construccion de vocabulario")
    #build_vocabulary_from_clean()
    #print("Tercera fase: Ingeniera de caracteristicas")
    #run_feature_pipeline()
    #print("Cuarta fase: creacion de clusters")
    #run_clustering_pipeline()
if __name__ == "__main__":
    main()
