'''
analysis/
---------
Bloque 8 del pipeline NLP: análisis semántico y estadístico avanzado.

Módulos:
    sentiment_analysis  -- sentimiento por estrellas, tópico y destino
    entity_analysis     -- análisis basado en entidades NER
    cooccurrence_graph  -- co-ocurrencia de entidades y términos
    trend_detection     -- distribución de tópicos, tendencias y perfiles
    analysis_pipeline   -- orquestador principal, punto de entrada del bloque
'''

from analysis.analysis_pipeline import run_analysis_pipeline

__all__ = ['run_analysis_pipeline']