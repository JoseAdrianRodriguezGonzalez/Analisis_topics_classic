import pandas as pd
from preprocessing.individual_functions import *
from preprocessing.processing_pipe import process_pipeline
#from semantic_expression.pipeline import pipe
from feature_engineering.vocabulary import build_vocabulary_from_clean
from feature_engineering.features import run_feature_pipeline



def preprocessing_complete(input_file):
    create_data_folders()
    spanish, english, mixed = process_pipeline(input_file)
    save_results(spanish, "data_spanish")
    save_results(english, "data_english")
    if len(mixed) > 0:
        pd.DataFrame(mixed).to_csv("data/data_mixed/mixed.csv", index=False)
        print("⚠️  File generated for mixed results.")
    print("\nPipeline executed successfully.")
def main():
    preprocessing_complete("/home/jorge/6to/Visualizacion de la informacion/Turismo/huatulco.csv")
    #pipe()
    build_vocabulary_from_clean()
    run_feature_pipeline()
if __name__ == "__main__":
    main()
