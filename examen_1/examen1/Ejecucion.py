import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modeloDifuso import ClasificadorBebidas

# Configuración
INPUT_FILE = 'vectors_wide.csv'
OUTPUT_FILE = 'Resultados_Difusos.csv'
OUTPUT_PLOT = 'Mapa_Difuso_Bebidas.png'

# Polos (Igual que antes)
POLES = {
    'Pulque':  np.array([-1,  1]),
    'Sake':    np.array([ 1,  1]),
    'Sidra':   np.array([ 1, -1]),
    'Tepache': np.array([-1, -1])
}

def calcular_coordenada_final(scores):
    final_x, final_y = 0, 0
    total_score = sum(scores.values())
    
    if total_score == 0: return 0, 0 # Centro si nadie vota
    
    for drink, score in scores.items():
        weight = score / total_score
        pole = POLES[drink]
        final_x += pole[0] * weight
        final_y += pole[1] * weight
        
    return final_x, final_y

def main():
    df = pd.read_csv(INPUT_FILE)
    motor = ClasificadorBebidas()
    
    results = []
    print(f"Procesando {len(df)} sesiones...")
    
    for _, row in df.iterrows():
        # Ejecución del Modelo
        res = motor.predecir(
            row['Val_F1'], row['Aro_F1'],
            row['Val_F2'], row['Aro_F2'],
            row['Val_F3'], row['Aro_F3']
        )
        
        # Coordenadas Visuales
        fx, fy = calcular_coordenada_final(res)
        
        # Ganador
        winner = max(res, key=res.get)
        
        # Guardar
        out_row = row.to_dict()
        out_row.update({
            'Fuzzy_X': fx,
            'Fuzzy_Y': fy,
            'Prediccion_Difusa': winner,
            'Score_Pulque': res['Pulque'],
            'Score_Sake': res['Sake'],
            'Score_Sidra': res['Sidra'],
            'Score_Tepache': res['Tepache']
        })
        results.append(out_row)

    # Guardar CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_FILE, index=False)
    print(f"Resultados guardados en: {OUTPUT_FILE}")

    # --- GRAFICAR CON JITTER (RUIDO VISUAL) ---
    print("Generando mapa visual con Jitter...")
    plt.figure(figsize=(10, 10))
    
    # Fondo de Cuadrantes
    plt.axhline(0, color='black', alpha=0.3)
    plt.axvline(0, color='black', alpha=0.3)
    plt.fill_between([-1.2, 0], 0, 1.2, color='green', alpha=0.05)
    plt.fill_between([0, 1.2], 0, 1.2, color='blue', alpha=0.05)
    plt.fill_between([0, 1.2], -1.2, 0, color='orange', alpha=0.05)
    plt.fill_between([-1.2, 0], -1.2, 0, color='red', alpha=0.05)

    # APLICAR JITTER
    # Agregamos ruido aleatorio solo para la visualización
    jitter_strength = 0.08 # Ajusta este valor si quieres más/menos dispersión
    df_res['Jitter_X'] = df_res['Fuzzy_X'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df_res))
    df_res['Jitter_Y'] = df_res['Fuzzy_Y'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df_res))

    # Scatter Plot usando las coordenadas con Jitter
    sns.scatterplot(
        data=df_res, x='Jitter_X', y='Jitter_Y', 
        hue='drink', style='drink', 
        palette='bright', s=120, alpha=0.7, edgecolor='black'
    )
    
    # Etiquetas
    plt.text(-1.1, 1.1, 'PULQUE\n(Tristeza)', color='green', fontweight='bold')
    plt.text(1.0, 1.1, 'SAKE\n(Calma)', color='blue', fontweight='bold')
    plt.text(1.0, -1.1, 'SIDRA\n(Felicidad)', color='orange', fontweight='bold')
    plt.text(-1.1, -1.1, 'TEPACHE\n(Disgusto)', color='red', fontweight='bold')

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.title("Mapa de Clasificación Difusa (con Jitter visual)")
    plt.savefig(OUTPUT_PLOT)
    print(f"Gráfica generada: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()