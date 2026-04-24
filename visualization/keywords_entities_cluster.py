import pandas as pd
import plotly.express as px
from pathlib import Path

#Directorio base
BASE_DIR = Path(__file__).resolve().parent.parent

#Función para la generación de keyowrds entities
def generar_keywords_entities():
    print("Generando Keywords + Entities...")
    try:
        ruta_kw = BASE_DIR / 'data/topic_enrichment/embeddings/kmeans_k8/keywords_por_cluster.csv'
        df_kw = pd.read_csv(ruta_kw)

        df_top = df_kw[df_kw['rank'] <= 10].copy()
        df_top = df_top.sort_values(by=['cluster_id', 'score_tfidf'], ascending=[True, True])

        fig = px.bar(
            df_top,
            x='score_tfidf',
            y='termino',
            facet_col='cluster_id',
            facet_col_wrap=4,
            color='score_tfidf',
            color_continuous_scale='Viridis',
            title="Top Keywords y Entidades por Tópico (TF-IDF)",
            height=800
        )

        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(showlegend=False)

        output_file = BASE_DIR / "visualization" / "keywords_entities_2.html"
        fig.write_html(str(output_file))
        print(f"Gráfica generada: {output_file.name}")
    except Exception as e:
        print(f"Error al generar Keywords + Entities: {e}")

#Ejecución
if __name__ == "__main__":
    print("Generación de visualizaciones\n")
    generar_keywords_entities()
    print("\nProceso terminado")