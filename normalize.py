import pandas as pd
import unicodedata
import re
import math
import openpyxl

# ======================================== #
#              LOAD FILE DATA              #
# ======================================== #
# ======================================== #
#            NORMALIZE COLUMNS             #
# ======================================== #

# ======================================== #
#              NORMALIZATION               #
# ======================================== #
def normalize_text(text):
    if(pd.isna(text)):
        return text
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text
# ======================================== #
#           DELETE DUPLICTE EMAILS         #
# ======================================== #
# ======================================== #
#                CLEAN AGES                #
# ======================================== #

def clean_ages(value):
    if(pd.isna(value)):
        return None
    
    match = re.search(r"\d+\.?\d*", str(value))
    if(match):
        number = float(match.group())
        return math.floor(number)
    return None
# ======================================== #
#            FIX PLACE OF ORIGIN           #
# ======================================== #
municipios_guanajuato = [
    "abasolo","acambaro","apaseo el alto","apaseo el grande","atarjea",
    "celaya","comonfort","coroneo","cortazar","cueramaro",
    "doctor mora","dolores hidalgo","guanajuato","huanimaro","irapuato",
    "jaral del progreso","jerécuaro","leon","manuel doblado","moroleon",
    "ocampo","penjamo","pueblo nuevo","purisima del rincon",
    "romita","salamanca","salvatierra","san diego de la union",
    "san felipe","san francisco del rincon","san jose iturbide",
    "san luis de la paz","santa catarina","santiago maravatío",
    "silao","tarandacuao","tarimoro","tierra blanca",
    "uriangato","valle de santiago","victoria","villagran",
    "xichu","yuriria"
]
municipios_estados=["orizaba","michoacan","oaxaca","arandas","oaxaca"]
def fix_place_of_origin(value):
    if(pd.isna(value)):
        return "otro"

    text = str(value)
    municipios_guanajuato.extend(municipios_estados)
    for municipio in municipios_guanajuato:
        if(municipio in text):
            return municipio
    return "otro"
# ======================================== #
#            REMOVE LINE BREAKS            #
# ======================================== #
def remove_line_breaks(text):
    if(pd.isna(text)):
        return text
    return str(text).replace("\n", " ").replace("\r", " ").strip()
# ======================================== #
#           SAVE CLEANED DATA              #
# ======================================== #
def normalize_column_names(df, mapping):
    df=df.rename(columns=lambda c :c.strip().lower())
    df=df.rename(columns=mapping)
    cols=set({v for v in mapping.values()})
    return df[[c for c in cols if c in df.columns]]
def join_datasets(path_df_1,path_df_2):
    df_1=pd.read_csv(path_df_1)
    df_2=pd.read_csv(path_df_1)
    results={
        "de que lugar procedes?": "lugar",
        "cual es tu edad?":"edad",
        "edad":"edad",
        "ciudad de origen":"lugar",
        "con que genero te identificas":"genero",
        "genero":"genero",
        "en tus propias palabras, que opinas de salamanca y que palabra, frase o recuerdo usarias para describirla principalmente?":"comentario",
        "en el siguiente espacio, expresa tu opinion general acerca de salamanca":"comentario"
    }
    df_1=normalize_column_names(df_1,results)
    df_2=normalize_column_names(df_2,results)
    df_final=pd.concat([df_1,df_2],ignore_index=True)
    return df_final

def pipeline_normalize_raw_data():
    print("---"*18)
    print("# Conjunto de datos del primer instrumento #")
    print("---"*18)
    df = pd.read_excel("data/raw/data_956.xlsx")
    df.columns = (
        df.columns.str.strip().str.lower().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    print("Step 1 completed: Normalize columns")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(normalize_text)
    print("Step 2 completed: Normalization")
    df = df.drop_duplicates(subset=["correo electronico"])
    print("Step 3 completed: Delete duplicate emails")
    df["cual es tu edad?"] = df["cual es tu edad?"].apply(clean_ages) 
    print("Step 4 completed: Clean ages")
    df["de que lugar procedes?"] = df["de que lugar procedes?"].apply(fix_place_of_origin)
    print("Step 5 completed: Fix place of origin")
    df["en el siguiente espacio, expresa tu opinion general acerca de salamanca"] = df["en el siguiente espacio, expresa tu opinion general acerca de salamanca"].apply(remove_line_breaks)
    print("Step 6 completed: Remove line breaks")
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Step 7 completed: Save cleaned data")
    print("---"*21)
    print("# Conjunto de datos del segundo instrumento instrumento #")
    print("---"*21)
    df = pd.read_csv("data/raw/data_survey.csv")
    df.columns = (
        df.columns.str.strip().str.lower().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    print("Step 1 completed: Normalize columns")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(normalize_text)
    print("Step 2 completed: Normalization")
    #df = df.drop_duplicates(subset=["correo electronico"])
    #print("Step 3 completed: Delete duplicate emails")
    df["edad"] = df["edad"].apply(clean_ages) 
    print("Step 3 completed: Clean ages")
    df["ciudad de origen"] = df["ciudad de origen"].apply(fix_place_of_origin)
    print("step 4 completed: fix place of origin")
    print(df.columns)
    df["en tus propias palabras, que opinas de salamanca y que palabra, frase o recuerdo usarias para describirla principalmente?"] = df["en tus propias palabras, que opinas de salamanca y que palabra, frase o recuerdo usarias para describirla principalmente?"].apply(remove_line_breaks)
    print("Step 5 completed: Remove line breaks")
    df.to_csv("data/processed/cleaned_data_survey_2.csv", index=False)
    print("Step 6 completed: Save cleaned data")
    print("---"*21)
    print("# Uniendo conjunto de datos y normalizando columnas #")
    print("---"*21)
    df_final=join_datasets("data/processed/cleaned_data.csv","data/processed/cleaned_data_survey_2.csv")
    df_final.to_csv("data/processed/data_basis.csv")
