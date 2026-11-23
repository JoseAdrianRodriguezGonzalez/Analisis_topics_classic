import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import sys

# Configuración para que pandas imprima todo sin cortar columnas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    try:
        df = pd.read_csv('Resultados_Difusos.csv')
    except:
        print("❌ ERROR: No encuentro Resultados_Difusos.csv. Ejecuta primero Ejecucion.py")
        return

    # 1. PREPARACIÓN DE DATOS
    y_true = df['drink']
    y_pred = df['Prediccion_Difusa']
    labels = sorted(df['drink'].unique())

    # 2. MÉTRICAS CLÁSICAS (HARD)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    # 3. PROCESO DE EVALUACIÓN DE DESDIFUSIFICACIÓN (Top-1 y Top-2)
    top1_hits = 0
    top2_hits = 0
    confidence_in_truth = []
    margins = []
    
    # Acumuladores para análisis
    scores_by_drink = {label: [] for label in labels}

    print("\n" + "="*60)
    print(f"   REPORTE DE DESDIFUSIFICACIÓN Y EXACTITUD")
    print("="*60 + "\n")
    
    # Encabezado de tabla de monitoreo
    print(f"{'REAL':<10} | {'PREDICCIÓN (Top-1)':<20} | {'SCORES DESDIFUSIFICADOS (Crisp Values)':<40} | {'¿TOP-2?'}")
    print("-" * 90)

    for index, row in df.iterrows():
        verdad = row['drink']
        
        # 1. OBTENCIÓN DE VALORES CRISP
        scores = {
            'Pulque': row['Score_Pulque'],
            'Sake': row['Score_Sake'],
            'Sidra': row['Score_Sidra'],
            'Tepache': row['Score_Tepache']
        }
        
        scores_by_drink[verdad].append(scores[verdad])
        
        # 2. RANKING
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_scores[0][0]
        runner_up = sorted_scores[1][0]
        
        # 3. EVALUACIÓN
        is_top1 = (verdad == winner)
        if is_top1:
            top1_hits += 1

        is_top2 = (verdad == winner or verdad == runner_up)
        if is_top2:
            top2_hits += 1
            
        confidence_in_truth.append(scores[verdad])
        if not is_top1:
            margins.append(scores[winner] - scores[verdad])

        # Preparamos los textos visuales
        scores_str = f"1º {winner}({scores[winner]:.0f})  2º {runner_up}({scores[runner_up]:.0f})"
        
        # Estado: Acierto, Casi (2do lugar), Fallo
        if is_top1:
            status = "ACIERTO"
        elif is_top2:
            status = "CASI (Top-2)"
        else:
            status = "FALLO"

        # Imprimimos TODO. Si quieres filtrar, agrega un if aquí.
        print(f"{verdad:<10} | {winner:<20} | {scores_str:<40} | {status}")

    # CÁLCULO FINAL DE METRICAS
    acc_top1 = top1_hits / len(df)
    acc_top2 = top2_hits / len(df)
    avg_confidence = np.mean(confidence_in_truth)
    avg_defeat_margin = np.mean(margins) if margins else 0

    # --- IMPRESIÓN DE RESULTADOS FINALES ---
    print("-" * 90)
    print("\n--- RESUMEN DE EXACTITUD ---")
    print(f"1. EXACTITUD ESTÁNDAR (Top-1):  {acc_top1:.2%}  (Acierto exacto)")
    print(f"2. EXACTITUD DIFUSA (Top-2):    {acc_top2:.2%}  (La respuesta correcta estaba entre las 2 mejores)")
    print(f"   -> Ganancia de Inteligencia: +{(acc_top2 - acc_top1):.2%} al considerar la segunda opción.")
    print("\n")

    print("--- 3. ANÁLISIS DE CONFIANZA (Promedio de Score obtenido) ---")
    for drink in labels:
        if scores_by_drink[drink]:
            avg_score = np.mean(scores_by_drink[drink])
            print(f"  - Cuando es {drink}, el modelo le da: {avg_score:.1f} / 100 pts")
        else:
            print(f"  - {drink}: Sin datos")
    
    print("\n" + "="*60)

    # 4. GENERACIÓN DE GRÁFICAS (PNG)
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Matriz Confusión
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusión\n(Accuracy: {acc:.1%})')
    plt.xlabel('Predicción')
    plt.ylabel('Real')

    # Plot 2: Barras de Aciertos vs Fallos
    plt.subplot(1, 2, 2)
    # Preparamos datos para barra apilada
    correct_counts = np.diag(cm)
    total_counts = np.sum(cm, axis=1)
    incorrect_counts = total_counts - correct_counts
    
    bar_width = 0.6
    indices = np.arange(len(labels))
    
    p1 = plt.bar(indices, correct_counts, bar_width, color='green', label='Aciertos', alpha=0.7)
    p2 = plt.bar(indices, incorrect_counts, bar_width, bottom=correct_counts, color='red', label='Fallos', alpha=0.7)
    
    plt.xticks(indices, labels)
    plt.title('Rendimiento por Bebida\n(Verde es bueno, Rojo es malo)')
    plt.legend()
    
    # Añadir etiquetas de porcentaje
    for i, (corr, tot) in enumerate(zip(correct_counts, total_counts)):
        if tot > 0:
            perc = corr / tot * 100
            plt.text(i, tot + 1, f"{perc:.0f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Diagnostico_Inteligente.png')
    print(f"✅ Gráfica guardada en: Diagnostico_Inteligente.png")

if __name__ == "__main__":
    main()