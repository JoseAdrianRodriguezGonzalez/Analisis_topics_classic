import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os
import re
import json

#Directio/Rutas
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))  # carpeta llm_model/
DIR_RAIZ = os.path.dirname(DIR_SCRIPT)
DIR_PROMPTS = os.path.join(DIR_RAIZ, "data", "analisis_clusters", "prompts")
os.makedirs(DIR_PROMPTS, exist_ok=True)
DIR_SALIDA = DIR_SCRIPT

print(f"Buscando prompts en: {DIR_PROMPTS}")

#Configuración del modelo Qwen 2.5 (7B)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="cuda",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

generador = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)


def procesar_texto_descriptores(nombre_archivo_txt, nombre_salida_csv):
    ruta_txt = os.path.join(DIR_PROMPTS, nombre_archivo_txt)

    if not os.path.exists(ruta_txt):
        print(f"No se encontró {nombre_archivo_txt}")
        return

    with open(ruta_txt, 'r', encoding='utf-8') as f:
        contenido = f.read()

    # Separar el contenido por clústeres
    bloques = re.split(r'Cluster (\d+)', contenido)
    resultados_llm = []

    # Iterar sobre los bloques detectados
    for i in range(1, len(bloques), 2):
        id_cluster = bloques[i]
        datos_cluster = bloques[i + 1].split('=' * 50)[0].strip()

        print(f"Analizando descriptores estadísticos del Clúster {id_cluster}...")

        prompt_ia = f"""Analiza las siguientes palabras clave y sus métricas TF-IDF que definen al Clúster {id_cluster}:
{datos_cluster}

Genera un análisis técnico y devuelve ÚNICAMENTE un JSON con esta estructura:
{{
  "sintesis": "Un párrafo breve del tema principal",
  "interpretacion": "La percepción o queja ciudadana que proyectan estos términos",
  "etiquetas": "4 palabras clave separadas por comas"
}}"""

        mensajes = [
            {"role": "system",
             "content": "Eres un analista de datos experto. Solo respondes con JSON válido sin texto adicional."},
            {"role": "user", "content": prompt_ia}
        ]

        full_prompt = tokenizer.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)

        salida = generador(
            full_prompt,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True
        )
        texto_generado = salida[0]['generated_text'].strip()

        try:
            # Extraer solo el contenido entre llaves por si el modelo añade texto extra
            match = re.search(r'\{.*\}', texto_generado, re.DOTALL)
            if match:
                datos_json = json.loads(match.group(0))
                resultados_llm.append({
                    'Cluster': int(id_cluster),
                    'Sintesis_Tema': datos_json.get("sintesis", ""),
                    'Interpretacion_Semantica': datos_json.get("interpretacion", ""),
                    'Etiquetas_Conceptuales': datos_json.get("etiquetas", "")
                })
        except Exception as e:
            print(f"Error en JSON de clúster {id_cluster}: {e}")

    # Guardar en CSV
    if resultados_llm:
        df_final = pd.DataFrame(resultados_llm)
        ruta_csv_final = os.path.join(DIR_SALIDA, nombre_salida_csv)
        df_final.to_csv(ruta_csv_final, index=False)
        print(f"CSV generado: {nombre_salida_csv}")

#Ejecución
if __name__ == "__main__":
    tareas = [
        ("top5_unigramas_jerarquico.txt", "llm_unigramas_jerarquico.csv"),
        ("top5_bigramas_jerarquico.txt", "llm_bigramas_jerarquico.csv"),
        ("top5_bigramas_dbscan.txt", "llm_bigramas_dbscan.csv"),
        ("top5_trigramas_jerarquico.txt", "llm_trigramas_jerarquico.csv")
    ]

    for txt, csv in tareas:
        procesar_texto_descriptores(txt, csv)
