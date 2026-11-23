import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

def analyze_data():
    print("--- INICIANDO ANÁLISIS DE DATOS (DEBUG PROFUNDO) ---")
    
    # 1. Cargar Datos
    try:
        df_wide = pd.read_csv('vectors_wide.csv')
        print(f"Archivo 'vectors_wide.csv' cargado. Dimensiones: {df_wide.shape}")
    except FileNotFoundError:
        print("ERROR: No se encuentra vectors_wide.csv")
        return

    # 2. Análisis de Escala (Universos Difusos)
    # ¿De verdad necesitamos un rango de -5 a +5?
    features = ['Val_F1', 'Aro_F1', 'Val_F2', 'Aro_F2', 'Val_F3', 'Aro_F3']
    print("\n--- 1. RANGOS REALES DE LOS DATOS (Min/Max Global) ---")
    global_stats = df_wide[features].agg(['min', 'max', 'mean', 'std']).round(2)
    print(global_stats)
    
    # 3. La "Firma" de cada Bebida
    print("\n--- 2. PROMEDIOS POR BEBIDA (La 'Verdad' Estadística) ---")
    means = df_wide.groupby('drink')[features].mean().round(2)
    print(means)

    # 4. Análisis de Variabilidad (¿Qué tan dispersos están?)
    # Si la Std Dev es alta (>1.0), las reglas estrictas fallarán.
    print("\n--- 3. DESVIACIÓN ESTÁNDAR (¿Qué tan caóticos son los usuarios?) ---")
    stds = df_wide.groupby('drink')[features].std().round(2)
    print(stds)

    # 5. Análisis de Deltas (Dinámica Temporal)
    # Fundamental para reglas de "Cambio"
    df_wide['Delta_Inicio_Medio'] = df_wide['Val_F2'] - df_wide['Val_F1']
    df_wide['Delta_Medio_Fin'] = df_wide['Val_F3'] - df_wide['Val_F2']
    df_wide['Delta_Total'] = df_wide['Val_F3'] - df_wide['Val_F1']
    
    deltas = ['Delta_Inicio_Medio', 'Delta_Medio_Fin', 'Delta_Total']
    print("\n--- 4. DINÁMICA TEMPORAL (¿Sube o Baja?) ---")
    delta_stats = df_wide.groupby('drink')[deltas].mean().round(2)
    print(delta_stats)

    # 6. ¿Quién se parece a quién? (Distancia Euclidiana)
    # Si la distancia es pequeña (< 1.0), es matemáticamente imposible separarlos linealmente.
    print("\n--- 5. MATRIZ DE CONFUSIÓN POTENCIAL (Distancia entre Promedios) ---")
    drinks = means.index
    print(f"{'Bebida A':<10} {'Bebida B':<10} {'Distancia':<10} {'Conclusión'}")
    print("-" * 50)
    
    for i, d1 in enumerate(drinks):
        for j, d2 in enumerate(drinks):
            if i < j: # Solo triangular superior para no repetir
                # Calculamos distancia usando las 6 dimensiones
                dist = euclidean(means.loc[d1], means.loc[d2])
                
                conclusion = "DIFERENTES"
                if dist < 1.0: conclusion = "¡GEMELOS! (Peligro)"
                elif dist < 2.0: conclusion = "SIMILARES"
                
                print(f"{d1:<10} {d2:<10} {dist:.2f}       {conclusion}")

if __name__ == "__main__":
    analyze_data()