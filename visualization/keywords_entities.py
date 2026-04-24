"""
Visualización 2 — Keywords + Entidades por Destino

Panel 2×5:
    Fila 1 : top-5 tópicos por destino (barras horizontales)
    Fila 2 : top-8 entidades por destino, coloreadas por sentimiento medio
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
from matplotlib.colors import Normalize

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_OUT = os.path.join(ROOT,"visualization")
os.makedirs(DIR_OUT, exist_ok=True)

PATH_TEND = os.path.join(ROOT, "data", "analysis", "trends", "tendencias_topicos_destino.csv")
PATH_ENT  = os.path.join(ROOT, "data", "analysis", "entities", "entidades_por_destino.csv")

DEST_ORDER  = ["huatulco", "la_paz", "riviera_nayarit", "puerto_vallarta", "riviera_maya"]
DEST_LABELS = ["Huatulco", "La Paz", "Riviera Nayarit", "Puerto Vallarta", "Riviera Maya"]
DEST_COLORS = ["#0097A7", "#388E3C", "#F57C00", "#7B1FA2", "#C62828"]

def clean_topic_name(raw):
    parts = str(raw).split("_")
    clean = [p for p in parts[1:] if p and not p.startswith("ent")]
    label = " ".join(clean[:4])
    return label.capitalize() if label else raw

def run_keywords_entities():
    df_tend = pd.read_csv(PATH_TEND, encoding="utf-8-sig")
    df_ent  = pd.read_csv(PATH_ENT,  encoding="utf-8-sig")
    df_tend["label"] = df_tend["topic_name"].apply(clean_topic_name)

    fig = plt.figure(figsize=(20, 13), facecolor="white")
    fig.suptitle("Keywords & Entidades por Destino Turístico",
                 fontsize=18, fontweight="bold", color="#1a1a2e", y=0.97)

    gs = fig.add_gridspec(2, 5, hspace=0.60, wspace=0.50,
                          top=0.93, bottom=0.05, left=0.06, right=0.96)

#Top 5 topicos
    for col, (dest, dlabel, dcolor) in enumerate(
            zip(DEST_ORDER, DEST_LABELS, DEST_COLORS)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#f7f9fc")

        sub    = df_tend[df_tend["location"] == dest].nlargest(5, "n_docs")
        lbls   = [l[:24]+"…" if len(l)>24 else l for l in sub["label"].tolist()]
        values = sub["n_docs"].tolist()

        ax.barh(range(len(values)), values, color=dcolor, alpha=0.82, height=0.6)
        ax.set_yticks(range(len(lbls)))
        ax.set_yticklabels(lbls, fontsize=7.5, color="#333")
        ax.invert_yaxis()
        ax.set_xlabel("# docs", fontsize=8, color="#777")
        ax.set_title(dlabel, fontsize=11, fontweight="bold", color=dcolor, pad=7)
        ax.tick_params(axis="x", colors="#aaa", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#ddd")
        ax.xaxis.grid(True, color="#ebebeb", linewidth=0.5)
        ax.set_axisbelow(True)
        mx = max(values) if values else 1
        for i, v in enumerate(values):
            ax.text(v + mx*0.02, i, str(v), va="center", fontsize=6.5, color="#666")

#Top-8 entidades coloreadas por sentimiento
    norm = Normalize(vmin=-1, vmax=1)
    cmap = cmaps.get_cmap("RdYlGn")  # noqa

    for col, (dest, dlabel) in enumerate(zip(DEST_ORDER, DEST_LABELS)):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#f7f9fc")

        sub    = df_ent[df_ent["location"] == dest].nlargest(8, "n_documentos")
        ents   = [e[:20]+"…" if len(e)>20 else e for e in sub["entidad"].tolist()]
        counts = sub["n_documentos"].tolist()
        sents  = sub["sentimiento_medio"].fillna(0).tolist()
        bar_colors = [cmap(norm(s)) for s in sents]

        ax.barh(range(len(counts)), counts, color=bar_colors, alpha=0.88, height=0.65)
        ax.set_yticks(range(len(ents)))
        ax.set_yticklabels(ents, fontsize=7.5, color="#333")
        ax.invert_yaxis()
        ax.set_xlabel("# docs", fontsize=8, color="#777")
        ax.set_title(f"{dlabel}\n(entidades)", fontsize=10,
                     fontweight="bold", color="#333", pad=5)
        ax.tick_params(axis="x", colors="#aaa", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#ddd")
        ax.xaxis.grid(True, color="#ebebeb", linewidth=0.5)
        ax.set_axisbelow(True)
        mx = max(counts) if counts else 1
        for i, v in enumerate(counts):
            ax.text(v + mx*0.02, i, str(v), va="center", fontsize=6.5, color="#666")

#Colorbar sentimiento
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.965, 0.06, 0.013, 0.37])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Sentimiento\nmedio", fontsize=8, color="#333")
    cb.ax.yaxis.set_tick_params(color="#555", labelcolor="#333", labelsize=7)

    out = os.path.join(DIR_OUT, "keywords_entities.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"keywords_entities.png  =  {out}")

if __name__ == "__main__":
    run_keywords_entities()