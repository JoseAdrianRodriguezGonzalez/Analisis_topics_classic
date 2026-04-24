import requests
import time
import logging
from llama_cpp import  Llama
llm=Llama(
    model_path="models/mistral-7b-instruct-v0.3-q4_k_m.gguf",
    n_ctx=1024,
    n_threads=8,
    n_gpu_layers=20
)
logger = logging.getLogger(__name__)
def  build_prompt(keywords,docs):
    keywords_str = ", ".join(
        f"{kw['termino']} ({kw.get('score_tfidf', 0):.2f})"
        if isinstance(kw, dict)
        else str(kw)
        for kw in keywords[:15]
    )
    docs_str = "\n".join(d.get("text", str(d)) 
        if isinstance(d, dict) else str(d)
        for d in docs[:3])
    prompt = f"""
    [INST]
    Eres experto en análisis de tópicos.

    Genera un nombre corto (máximo 5 palabras).

    Keywords: {keywords_str}
    Docs: {docs_str}

    Responde SOLO con el nombre.
    [/INST]
    """
    return prompt.strip()
def query_mistral_local(prompt, max_tokens=20):
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["\n", "Topic:", "Cluster:"]
        )
        text = output["choices"][0]["text"].strip()
        return text
    except Exception as e:
        logger.error(f"Error en modelo local: {e}")
        return "unknown_topic"
def name_all_clusters(keywords_por_cluster,docs_por_cluster):
    cluster_names={}
    for cluster_id in keywords_por_cluster.keys():
        keywords = keywords_por_cluster.get(cluster_id,[])
        docs=docs_por_cluster.get(cluster_id,[])
        if not keywords:
            cluster_names[cluster_id]="unknown"
            continue
        prompt =build_prompt(keywords,docs)
        name= query_mistral_local(prompt)
        name=name.replace("\n"," ").strip()
        cluster_names[cluster_id]=name 
        logger.info(f"Cluster: {cluster_id} -> {name}")
def name_single_cluster(keywords,docs):
    prompt=build_prompt(keywords,docs)
    return query_mistral_local(prompt)
