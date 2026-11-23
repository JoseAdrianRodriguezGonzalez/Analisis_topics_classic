import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ClasificadorBebidas:
    def __init__(self):
        print("   [INFO] Inicializando Motor Difuso (v35 - DIAGNÓSTICO DIFERENCIAL)...")

        # 1. ANTECEDENTES
        self.val_range = np.arange(-5, 5.1, 0.1) 
        self.aro_range = np.arange(-4, 4.1, 0.1)
        self.delta_range = np.arange(-5, 5.1, 0.1)

        self.delta_T = ctrl.Antecedent(self.delta_range, 'delta_Total') 
        self.delta_12 = ctrl.Antecedent(self.delta_range, 'delta_1to2') # NUEVO: Para medir el golpe inicial
        self.val_f3 = ctrl.Antecedent(self.val_range, 'val_f3') 
        self.aro_f1 = ctrl.Antecedent(self.aro_range, 'aro_f1')         

        # 2. CONSECUENTES
        self.score_range = np.arange(0, 101, 1)
        self.pulque = ctrl.Consequent(self.score_range, 'pulque', defuzzify_method='centroid')
        self.sake = ctrl.Consequent(self.score_range, 'sake', defuzzify_method='centroid')
        self.sidra = ctrl.Consequent(self.score_range, 'sidra', defuzzify_method='centroid')
        self.tepache = ctrl.Consequent(self.score_range, 'tepache', defuzzify_method='centroid')

        # 3. FUNCIONES DE MEMBRESÍA

        # --- SIDRA: Ganancia Neta ---
        # Delta Total positivo > 0.2
        self.delta_T['gain'] = fuzz.trapmf(self.delta_range, [0.2, 0.6, 5, 5])

        # --- TEPACHE: El Golpe Inicial (Crash) ---
        # Caída fuerte entre F1 y F2. Promedio real es -0.87.
        # Definimos "Crash" si cae más de -0.7.
        self.delta_12['crash'] = fuzz.trapmf(self.delta_range, [-5, -5, -1.5, -0.7])

        # --- SAKE: Calma y Estabilidad ---
        # Arousal negativo.
        self.aro_f1['zen'] = fuzz.trapmf(self.aro_range, [-5, -5, -0.2, 0.05])
        # Caída inicial SUAVE (No Crash). Entre -0.6 y 0.0.
        self.delta_12['soft'] = fuzz.trimf(self.delta_range, [-0.7, -0.3, 0.1])

        # --- PULQUE: Pérdida Total ---
        # Delta Total negativo < -0.4
        self.delta_T['loss'] = fuzz.trapmf(self.delta_range, [-5, -5, -1.0, -0.4])
        # Final Malo (Soporte)
        self.val_f3['bad'] = fuzz.trapmf(self.val_range, [-5, -5, -0.8, -0.3])

        # --- SALIDAS ---
        for s in [self.pulque, self.sake, self.sidra, self.tepache]:
            s['hit'] = fuzz.trimf(self.score_range, [50, 100, 100])

        # 4. REGLAS DIFERENCIALES
        reglas = []

        # R1: SIDRA (Prioridad: Mejora Global)
        reglas.append(ctrl.Rule(self.delta_T['gain'], self.sidra['hit']))

        # R2: TEPACHE (Prioridad: Golpe Inicial Fuerte)
        # Si hubo un crash inicial Y NO es una mejora global (Sidra).
        # Esto captura el "Valle" pero mirándolo como cambio, no como valor absoluto.
        reglas.append(ctrl.Rule(
            self.delta_12['crash'] & ~self.delta_T['gain'], 
            self.tepache['hit']
        ))

        # R3: SAKE (Prioridad: Sutileza)
        # Es Zen O tuvo una caída suave inicial.
        # Protegido contra Crash (Tepache) y Gain (Sidra).
        reglas.append(ctrl.Rule(
            (self.aro_f1['zen'] | self.delta_12['soft']) & ~self.delta_12['crash'] & ~self.delta_T['gain'], 
            self.sake['hit']
        ))

        # R4: PULQUE (Prioridad: Declive)
        # Pérdida total Y Final malo.
        # Protegido contra Crash fuerte (Tepache suele tener crashes más fuertes).
        reglas.append(ctrl.Rule(
            self.delta_T['loss'] & self.val_f3['bad'] & ~self.delta_12['crash'], 
            self.pulque['hit']
        ))
        
        # R5: PULQUE DEFAULT
        # Si todo falla, asumimos Pulque (la clase "triste por defecto").
        reglas.append(ctrl.Rule(
            ~self.delta_T['gain'] & ~self.delta_12['crash'] & ~self.aro_f1['zen'],
            self.pulque['hit']
        ))

        self.system = ctrl.ControlSystem(reglas)
        self.simulacion = ctrl.ControlSystemSimulation(self.system)

    def predecir(self, v_f1, a_f1, v_f2, a_f2, v_f3, a_f3, debug=False):
        d_total = v_f3 - v_f1
        d_1to2 = v_f2 - v_f1 # NUEVO CÁLCULO
        
        inputs = {
            'delta_1to2': np.clip(d_1to2, -4.9, 4.9),
            'val_f3': np.clip(v_f3, -4.9, 4.9),
            'aro_f1': np.clip(a_f1, -3.9, 3.9),
            'delta_Total': np.clip(d_total, -4.9, 4.9)
        }
        
        for key, value in inputs.items():
            self.simulacion.input[key] = value

        try:
            self.simulacion.compute()
            return {
                'Pulque': self.simulacion.output.get('pulque', 0),
                'Sake': self.simulacion.output.get('sake', 0),
                'Sidra': self.simulacion.output.get('sidra', 0),
                'Tepache': self.simulacion.output.get('tepache', 0)
            }
        except:
            return {'Pulque':25, 'Sake':25, 'Sidra':25, 'Tepache':25}