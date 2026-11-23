
from pandas.io.formats import printing
from scipy.cluster import hierarchy
from sqlalchemy import desc
from descriptivos import DA
from preprocessing import pipeline_preprocessing
import pandas as pd 
from variables_ import setVariables
from model import RegressionModel,GenderAnalysis,AgeAnalysis,LocationAnalysis,OriginAnalysis,cluster_kmean,Hierarchical,dbsc,visualization# Ejemplo
not_item_columns = [
    "marca temporal",
    "edad",
    "genero",
    "ciudad de origen",
    "ciudad de origen.1",
    "columna 23",
    "en tus propias palabras, que opinas de salamanca y que palabra, frase o recuerdo usarias para describirla principalmente?",
"En tus propias palabras, ¿qué opinas de Salamanca y qué palabra, frase o recuerdo usarías para describirla principalmente?"
]
def string_to_numerical(df:pd.DataFrame,columns):
    names={
        "muy de acuerdo":5,
        "de acuerdo":4,
        "neutral":3,
        "en desacuerdo":2,
        "muy en desacuerdo":1,
        "muy positiva":5,
        "positiva":4,
        "neutral":3,
        "negativa":2,
        "muy negativa":1
    }
    df[columns]=df[columns].apply(lambda c: c.str.lower().str.strip().map(names))
    print("Cantidad de NaN por columna:")
    return df 
def analisis_descriptivo(name_path:str,analisis:pd.DataFrame):
    df=analisis.drop(columns=not_item_columns) 
    print(df.isna().sum())
    
    print("---"*18)
    print("Intercambio de string a numerico")
    df_items=df
    df_items_analizar=pd.concat([df_items])
    print(df_items.columns)
    final_dataframe=DA(df_items_analizar)
    results={column:final_dataframe.info_column(column) for column in final_dataframe.df.columns}
    df_metrics=pd.DataFrame(results)
    df_metrics.to_csv(name_path)
    print("---"*18)
    print("CSV generado con las métricas númericas por cada item ")
    print("---"*18)
    return final_dataframe
def plot_items(descriptive_analysis:DA,folder:str):
    #boxplots
    print("---"*18)
    print("Generando boxplots e histogramas por item")
    print("---"*18)
    print(descriptive_analysis.df.isna().sum())
    for column in descriptive_analysis.df.columns:
        if column not in not_item_columns:
            descriptive_analysis.plot_boxplots(column,f"{folder}/boxplots/{column.replace('/','_')}")
            descriptive_analysis.plot_histogram(column,save_path=f"{folder}/histogram/{column.replace('/','_')}")
    
    print("---"*18)
    print("Finalizado con exito")
    print("---"*18)
def analisis_variables(df:pd.DataFrame,original_df,save_path,output_path="results/2_analisis"):
    items=[col for col in df.columns if col not in not_item_columns ]
    print(len(items)) 
    print(items)
    dictionary_varaibles={
        "CV":items[0:8],
        "AV":items[8:16],
        "GV":items[16:20]
    }
    Variables=setVariables(df,dictionary_varaibles["AV"],dictionary_varaibles["CV"],dictionary_varaibles["GV"])
    print("---"*18)
    print("Construye el score")
    print("---"*18)
    Variables.get_all() 
    print("Guarda el score")
    df_final = pd.concat([Variables.df,original_df[not_item_columns]],axis=1)
    print("Guardando dataset final")
    df_final.to_csv("out/2/dataset_scores.csv", index=False)
    print("---"*18)
    print("---"*18)
    print("Calculo de descriptivos en scores")
    print("---"*18)
    descriptive_variables={
        k:Variables.get_info(k)  for k in dictionary_varaibles
    }
    print("Calculo de descriptivos en scores")
    print(f"Guardando en.... {output_path}")
    for k,dictionaries in descriptive_variables.items():
        df_items=pd.DataFrame([dictionaries])
        df_items.to_csv(f"{output_path}/{k}.csv")
        print(f"Guardado con exito la variable {k}")
    else:
        print("Se guardaron exitosamente")
    print(Variables.variables_score.df.isna().sum())
    print("---"*18)
    print("Genera los graficos de distribucion de histogramas de boxplots, en base al score por variable")
    print("---"*18)
    
    for k in dictionary_varaibles:

        Variables.plot_histo_variable(k,f"{save_path}/histogram/{k}")
        Variables.plot_boxplot_variable(k,f"{save_path}/boxplots/{k}")
    print("Se calculan los alfas por variable")
    print("---"*18)
    df_alpha=Variables.get_alpha()
    df_alpha.to_csv(f"{output_path}/cronbach_alpha.csv", index=False)
    print("Se calculan las correlaciones por items y por scores de variables")
    print("---"*18)
    correlations_items=Variables.correlations(not_item_columns)
    correlations_scores=Variables.correlations_scores()
    print("Guardando archivos ...")
    print("---"*18)
    Variables.save_correlations(correlations_items,"out/2/per_item_")
    Variables.save_correlations(correlations_scores,"out/2/per_score_")

    print("Calculando y guardando en csv los valores kmo y bartlett test por variable ")
    print("---"*18)
    factibility={
        k :  Variables.save_adequacy_tests(k,"out/2")
        for k in dictionary_varaibles
    }
    print("Haciendo analisis factorial para las 3 variables")
    print("---"*18)
    for variable_name,factorable in factibility.items():
        if not factorable:
            print(f"{variable_name}: No es adecuado para análisis factorial")
            continue  
        print(f"{variable_name}: Ejecutando análisis factorial\n Eigenvalos: ")
        eigen=Variables.eigen_analysis(variable_name,f"{save_path}/scree/{variable_name}")
        df_eigen = pd.DataFrame({
            "factor": range(1, len(eigen["eigenvalues"]) + 1),
            "eigenvalue": eigen["eigenvalues"],
            "cumulative_variance": eigen["cumulative_variance"]
        })
        df_eigen.to_csv(f"{output_path}/{variable_name}_eigenvalues.csv", index=False)
        print("Calculando el analisis factorial")
        fa_results=Variables.factorial_analysis(variable_name)
        df_loadings = pd.DataFrame(
            fa_results["loadings"],
            index=Variables.dictionary_name[variable_name]
        )
        df_loadings.to_csv(f"{output_path}/{variable_name}_factor_loadings.csv")
        variance = pd.DataFrame(fa_results["variance_explained"])
        variance.to_csv(f"{output_path}/{variable_name}_variance_explained.csv")
def tercer_analisis(df, output_path="results/3_analysis"):
    print("---"*18)
    print("TERCER ANALISIS: Comparaciones entre grupos")
    print("---"*18)
    variables = ["AV","CV","GV"]
    gender = GenderAnalysis(df,gender_col="con que genero te identificas")
    age = AgeAnalysis(df,age_col="edad")
    location = LocationAnalysis(df,location_col="ciudad de origen")
    location.build_location_group()
    print(df["location_group"].value_counts())
    origin = OriginAnalysis(df,city_col="ciudad de origen")
    age.build_age_groups()
    origin.build_origin_group()
    results = []
    analyses = {
        "gender": gender,
        "age": age,
        "location": location,
        "origin": origin
    }
    for analysis_name, analysis in analyses.items():
        print(f"\nAnalizando {analysis_name}")
        for var in variables:
            r = analysis.compare(var)
            results.append({
                "analysis": analysis_name,
                "variable": var,
                "test": r["test"],
                "statistic": r["statistic"],
                "p_value": r["p_value"] if (analysis_name not in ["gender"]) else 0,
                "effect_size":r["effect_size"] if  (analysis_name not in ["age","gender"]) else 0,
                "direction":r["direction"] if (analysis_name not in ["age","gender"]) else 0
            })
            if analysis_name in ["age", "gender"]:
                posthoc_res = analysis.posthoc(var)
                print(f"\nPost-hoc para {var} ({analysis_name}):\n", posthoc_res)
                if hasattr(posthoc_res, "data"):
                    df_posthoc = pd.DataFrame(posthoc_res.data[1:], columns=posthoc_res.data[0])
                    df_posthoc.to_csv(f"{output_path}/{analysis_name}_{var}_posthoc.csv", index=False)
                else:
                    posthoc_res.to_csv(f"{output_path}/{analysis_name}_{var}_posthoc.csv", index=False) 
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{output_path}/group_comparisons.csv", index=False)
    print("Resultados guardados en:")
    print(f"{output_path}/group_comparisons.csv")
    print("---"*18)
    print("Modelo de regresión")
    print("---"*18)
    reg = RegressionModel(df)
    reg.model()
    with open(f"{output_path}/regression_model.txt","w") as f:
        f.write(str(reg))
    print("Modelo de regresión guardado")
def cuarto_analisis(df,output_path="results/3_analysis",plots="figures/clusters"):
    df=cluster_kmean(df)
    df=Hierarchical(df)
    df=dbsc(df)
    visualization(df,"cluster_dbscan")
    visualization(df,"cluster_hier")
    visualization(df,"cluster_kmeans")

def main():
    print("---"*18)
    print("|\t\t Preprocesamiento\t\t|")
    print("---"*18)
    pipeline_preprocessing()
    df=pd.read_csv("cleaned_data_survey_2.csv") 
    print("---"*18)
    print("|\t\t Primer analisis\t\t|")
    print("|\tAnalisis de estadísticas descriptivas\t|")
    print("---"*18)
    descriptive_metrics_df=analisis_descriptivo("out/1/metricas_per_items.csv",df)
    plot_items(descriptive_metrics_df,"figures/descriptive_items")
    print("---"*18)
    print("|\t\t Segundo analisis\t\t|")
    print("|\tAnalisis de variables de constructos\t|")
    print("---"*18)
    analisis_variables(descriptive_metrics_df.df,df,save_path="figures/variable")  
    df_scores = pd.read_csv("out/2/dataset_scores.csv")
    tercer_analisis(df_scores)
    cuarto_analisis(df_scores)
if __name__=="__main__":
    main() 
