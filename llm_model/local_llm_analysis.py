import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os
import re
import json

#Directorio/Rutas
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_SCRIPT)
DIR_DATOS = os.path.join(DIR_RAIZ, "data", "analisis_clusters")
DIR_SALIDA = DIR_SCRIPT

print("Inicializando Qwen 2.5 (7B)")

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

#Función de análisis LLM
def analizar_cluster_local(archivo_entrada, archivo_salida):
    ruta_entrada = os.path.join(DIR_DATOS, archivo_entrada)
    ruta_salida = os.path.join(DIR_SALIDA, archivo_salida)

    if not os.path.exists(ruta_entrada):
        print(f"\nError: No se encontró el archivo {ruta_entrada}")
        return

    print(f"Analizando: {archivo_entrada}")

    df_rep = pd.read_csv(ruta_entrada)
    resultados_llm = []

    for cluster_id, grupo in df_rep.groupby('cluster'):
        respuestas = grupo['comentario'].dropna().tolist()[:10]
        texto_respuestas = "\n".join([f"- {r}" for r in respuestas])

        mensajes = [
            {"role": "system",
             "content": "Eres un analista de datos experto. Solo puedes responder con un objeto JSON válido. No escribas texto fuera del JSON."},
            {"role": "user",
             "content": f"Analiza estos textos del Clúster {cluster_id}:\n{texto_respuestas}\n\nDevuelve ÚNICAMENTE un JSON con esta estructura exacta:\n{{\n  \"sintesis\": \"Un párrafo resumiendo el tema principal\",\n  \"interpretacion\": \"Un párrafo interpretando el tono, queja o emoción de fondo\",\n  \"etiquetas\": \"4 palabras clave separadas por comas\"\n}}"}
        ]

        prompt = tokenizer.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)

        print(f"Procesando Clúster {cluster_id}...")

        salida = generador(
            prompt,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True
        )
        texto_generado = salida[0]['generated_text'].strip()

        #Extracción segura de JSON
        sintesis, interpretacion, etiquetas = "", "", ""
        try:
            match = re.search(r'\{.*\}', texto_generado, re.DOTALL)
            if match:
                datos_json = json.loads(match.group(0))
                sintesis = datos_json.get("sintesis", "")
                interpretacion = datos_json.get("interpretacion", "")
                etiquetas = datos_json.get("etiquetas", "")
            else:
                sintesis = "Error: La IA no generó el JSON correctamente."
        except json.JSONDecodeError:
            sintesis = "Error al leer el JSON. Texto crudo: " + texto_generado

        resultados_llm.append({
            'Cluster': cluster_id,
            'Sintesis_Tema': sintesis,
            'Interpretacion_Semantica': interpretacion,
            'Etiquetas_Conceptuales': etiquetas
        })

    # Guardar resultados específicos
    pd.DataFrame(resultados_llm).to_csv(ruta_salida, index=False)
    print(f"Análisis guardado con éxito en: {archivo_salida}")

#Ejecución por lotes
if __name__ == "__main__":
    #Diccionario con el archivo de entrada y el nombre que tendrá el archivo de salida
    archivos_a_procesar = [
        ("representativos_unigramas_jerarquico.csv", "llm_unigramas_jerarquico.csv"),
        ("representativos_bigramas_jerarquico.csv", "llm_bigramas_jerarquico.csv"),
        ("representativos_bigramas_dbscan.csv", "llm_bigramas_dbscan.csv"),
        ("representativos_trigramas_jerarquico.csv", "llm_trigramas_jerarquico.csv")
    ]

    for entrada, salida in archivos_a_procesar:
        analizar_cluster_local(entrada, salida)

    print("\nTodos los análisis cualitativos guradados")