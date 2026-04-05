import pandas as pd
from preprocessing.individual_functions import *
from preprocessing.processing_pipe import process_pipeline
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
    preprocessing_complete("data/raw/huatulco.csv")
if __name__ == "__main__":
    main()
