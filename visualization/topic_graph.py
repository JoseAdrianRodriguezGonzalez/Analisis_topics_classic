"""
Visualización 1 — Mapa de Tópicos
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_OUT  = os.path.join(ROOT,"visualization")
os.makedirs(DIR_OUT, exist_ok=True)

PATH_PROJ = os.path.join(ROOT, "data", "clustering", "embeddings", "proyeccion_2d.npy")
PATH_ETIQ = os.path.join(ROOT, "data", "clustering", "embeddings", "etiquetas_mejores.json")
PATH_KW   = os.path.join(ROOT, "data", "topic_enrichment", "embeddings",
                          "kmeans_k8", "keywords_por_cluster.csv")

def _check_paths():
    missing = [p for p in [PATH_PROJ, PATH_ETIQ, PATH_KW] if not os.path.exists(p)]
    if missing:
        print("❌  Archivos no encontrados:")
        for p in missing:
            print(f"    {p}")
        print(f"\n    Raíz detectada del proyecto: {ROOT}")
        print("    Asegúrate de que data/clustering/ exista dentro de esa carpeta.")
        sys.exit(1)

COLORS = ["#1565C0","#00838F","#2E7D32","#E65100",
          "#6A1B9A","#C62828","#F9A825","#4E342E"]
SAMPLE_N = 4000
SEED     = 42

def run_topic_graph():
    _check_paths()

    X = np.load(PATH_PROJ)
    with open(PATH_ETIQ, encoding="utf-8") as f:
        etiq = json.load(f)
    labels = np.array(etiq["kmeans|k=8"])

    df_kw = pd.read_csv(PATH_KW, encoding="utf-8-sig")
    cluster_names = {}
    for c in range(8):
        kws = df_kw[df_kw["cluster_id"] == c].head(3)["termino"].tolist()
        cluster_names[c] = " · ".join(kws)

    # muestra
    rng   = np.random.default_rng(SEED)
    per_c = SAMPLE_N // 8
    idx   = []
    for c in range(8):
        ci = np.where(labels == c)[0]
        idx.extend(rng.choice(ci, min(per_c, len(ci)), replace=False))
    idx = np.array(idx)
    xs = X[idx, 0];  ys = X[idx, 1];  ls = labels[idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")

    for c in range(8):
        m = ls == c
        ax.scatter(xs[m], ys[m], c=COLORS[c], s=6, alpha=0.55,
                   linewidths=0, rasterized=True)

    for c in range(8):
        ci = idx[ls == c]
        if len(ci) == 0:
            continue
        cx, cy = X[ci, 0].mean(), X[ci, 1].mean()
        short  = " / ".join(cluster_names[c].split(" · ")[:2])
        ax.annotate(
            f"C{c}: {short}", (cx, cy),
            fontsize=8.5, fontweight="bold", color="white", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=COLORS[c], ec="none", alpha=0.90),
        )

    ax.set_xlabel("UMAP 1", fontsize=11, color="#555")
    ax.set_ylabel("UMAP 2", fontsize=11, color="#555")
    ax.set_title("Mapa de Tópicos — Proyección UMAP  (KMeans k=8, n=54 735 docs)",
                 fontsize=14, fontweight="bold", color="#1a1a2e", pad=14)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")
    ax.grid(True, color="#e5e5e5", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=COLORS[c],
               label=f"C{c}: {cluster_names[c]}") for c in range(8)]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              framealpha=0.95, facecolor="white", edgecolor="#ccc")

    plt.tight_layout()
    out = os.path.join(DIR_OUT, "topic_graph.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"topic_graph.png  =  {out}")

if __name__ == "__main__":
    run_topic_graph()