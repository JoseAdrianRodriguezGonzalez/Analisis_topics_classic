import pandas as pd
import json
from preprocessing.processing_pipe import process_pipeline

def main():

    input_file = "raw_data/data.csv"

    results_1_2, results_3, json_data = process_pipeline(input_file)

    # ------------------- #
    #      Save CSV 1     #
    # ------------------- #
    df1 = pd.DataFrame(results_1_2)
    df1.to_csv("data/analysis.csv", index=False)

    # ------------------- #
    #      Save CSV 2     #
    # ------------------- #
    df2 = pd.DataFrame(results_3)
    df2.to_csv("data/clean.csv", index=False)

    # ------------------- #
    #      Save JSON      #
    # ------------------- #
    with open("data/analysis.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print("✅ Pipeline ejecutado correctamente")

if __name__ == "__main__":
    main()