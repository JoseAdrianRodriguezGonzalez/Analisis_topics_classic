from pathlib import Path
import pandas as pd
import networkx as nx
from pyvis.network import Network

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT    = BASE_DIR / "data" / "analysis" / "cooccurrence" / "coocurrencia_entidades.csv"
OUT_DIR  = BASE_DIR / "visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT   = OUT_DIR / "topic_graph_2.html"

#Nodos a exlucir
# Falsos positivos del NER: pronombres/adverbios clasificados como LOC/ORG, más ruido de Instagram (hashtags compuestos)
FALSE_POSITIVES = {
    "fuimos", "ademas", "llegamos", "encima", "ademas",
    "falto", "buena", "tome",
    # hashtags de Instagram sin valor semántico de lugar
    "bikiniselfie bikiniseksi bikinisummer summerbodyready",
    "lapazmx bikinishot",
    "visitacancun agencia gestion",
    "beachlife",
    "visithuatulco",
    "huatulcooaxaca huatulco",   # duplicado ruidoso de "huatulco"
    "bcs",                        # abreviatura poco informativa con la paz
}

#Paleta de colores
COMMUNITY_COLORS = [
    "#1565C0", "#2E7D32", "#E65100", "#6A1B9A",
    "#C62828", "#00838F", "#F9A825", "#4E342E",
    "#37474F", "#AD1457",
]

def run_topic_graph_interactivo():
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    df_clean = df[
        ~df["entidad_a"].isin(FALSE_POSITIVES) &
        ~df["entidad_b"].isin(FALSE_POSITIVES) &
        (df["co_ocurrencias"] > 5)
    ].copy()

    if df_clean.empty:
        print("Mal. Revisar los umbrales.")
        return

    print(f"  Aristas originales : {len(df)}")
    print(f"  Aristas filtradas  : {len(df_clean)}")

#Grafica
    G = nx.Graph()

    #Agregar aristas con atributos
    for _, row in df_clean.iterrows():
        G.add_edge(
            row["entidad_a"], row["entidad_b"],
            weight=int(row["co_ocurrencias"]),
            pmi=round(float(row["pmi"]), 2),
        )

    #Agregar frecuencia de nodo (doc_freq) para tamaño
    freq_map = {}
    for _, row in df_clean.iterrows():
        freq_map[row["entidad_a"]] = int(row["doc_freq_a"])
        freq_map[row["entidad_b"]] = int(row["doc_freq_b"])

    print(f"  Nodos en el grafo  : {G.number_of_nodes()}")
    print(f"  Aristas en el grafo: {G.number_of_edges()}")

    #Comunidades
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    net = Network(
        height="820px",
        width="100%",
        bgcolor="#ffffff", #fondo
        font_color="#222222",
        notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.05,
        damping=0.09,
    )

    # Agregar nodos con tamaño y color por comunidad
    min_freq = min(freq_map.values())
    max_freq = max(freq_map.values())

    for node in G.nodes():
        freq  = freq_map.get(node, 10)
        # Normalizar tamaño entre 12 y 45
        size  = 12 + 33 * (freq - min_freq) / max(max_freq - min_freq, 1)
        comm_idx = node_community.get(node, 0) % len(COMMUNITY_COLORS)
        color = COMMUNITY_COLORS[comm_idx]

        degree = G.degree(node)
        title  = (
            f"<b>{node}</b><br>"
            f"Frecuencia en corpus: {freq}<br>"
            f"Conexiones: {degree}"
        )

        net.add_node(
            node,
            label=node,
            size=float(size),
            color=color,
            title=title,
            font={"size": 13, "color": "#222222"},
        )

    # Agregar aristas con grosor proporcional
    max_cooc = df_clean["co_ocurrencias"].max()
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        width = 1 + 7 * (w / max_cooc)
        title = f"Co-ocurrencias: {w}<br>PMI: {data['pmi']}"
        net.add_edge(u, v, width=float(width), title=title,
                     color={"color": "#aaaaaa", "opacity": 0.7})

    legend_items = ""
    for i, comm in enumerate(communities):
        color = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        top_nodes = sorted(comm, key=lambda n: G.degree(n), reverse=True)[:3]
        legend_items += (
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{color};margin-right:5px"></span>'
            f'Comunidad {i+1}: {", ".join(top_nodes)}<br>'
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200}
      }
    }
    """)

    net.write_html(str(OUTPUT))
    html = OUTPUT.read_text(encoding="utf-8")
    legend_html = f"""
    <div style="position:fixed;top:16px;left:16px;background:rgba(255,255,255,0.93);
                padding:14px 18px;border-radius:10px;border:1px solid #ddd;
                font-family:sans-serif;font-size:12px;max-width:320px;z-index:9999;
                box-shadow:0 2px 8px rgba(0,0,0,0.12)">
      <b style="font-size:14px">Grafo de Co-ocurrencia de Entidades</b><br>
      <span style="color:#777;font-size:11px">
        Nodo: tamaño = frecuencia en corpus<br>
        Arista: grosor = co-ocurrencias<br>
        Color: comunidad detectada
      </span><br><br>
      {legend_items}
    </div> 
    """ #?
    html = html.replace("<body>", "<body>\n" + legend_html, 1)
    OUTPUT.write_text(html, encoding="utf-8")

    print(f"topic_graph_2.html =  {OUTPUT}")

if __name__ == "__main__":
    run_topic_graph_interactivo()