'''
clustering/
-----------
Bloque 6 del pipeline NLP: agrupamiento de documentos sobre embeddings.

Módulos:
    kmeans_clustering       -- grid search KMeans
    hierarchical_clustering -- grid search Clustering Jerárquico
    hdbscan_clustering      -- grid search HDBSCAN
    clustering_pipeline     -- orquestador principal, punto de entrada del bloque
'''

from clustering.clustering_pipeline import run_clustering_pipeline
from clustering.clustering_visualizacion import run_visualizacion

__all__ = ['run_clustering_pipeline', 'run_visualizacion']