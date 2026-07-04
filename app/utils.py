import numpy as np
import pandas as pd
import torch

def generate_iiot_sequence(seq_len=50, anomaly_type=None, noise_level=0.1, missing_rate=0.0):
    """
    Genera una secuencia temporal sintética de sensores IIoT con marcas de tiempo irregulares.
    Sensores simulados:
      - Sensor 1: Temperatura (Celsius) - Estado base: ~70C
      - Sensor 2: Vibración (G) - Estado base: ~1.0G
      - Sensor 3: Presión (Bar) - Estado base: ~2.0 Bar
      
    Parámetros:
      - seq_len: longitud de la secuencia temporal.
      - anomaly_type: None (Normal), 'spikes' (picos de vibración), 'thermal' (sobrecalentamiento), 'leak' (caída de presión).
      - noise_level: Desviación estándar del ruido gaussiano añadido.
      - missing_rate: Fracción de lecturas perdidas (NaN) para simular pérdida de paquetes.
    """
    # 1. Generar intervalos de tiempo irregulares (Delta t)
    # En lugar de dt = 1.0, simulamos intervalos irregulares muestreados de una distribución exponencial
    dt_intervals = np.random.exponential(scale=0.8, size=seq_len) + 0.2
    # El tiempo acumulativo (timestamps)
    timestamps = np.cumsum(dt_intervals)
    
    # 2. Generar señales base
    # Temperatura (señal lenta y estable)
    temp = 70.0 + np.sin(timestamps * 0.1) * 2.0
    # Vibración (alta frecuencia y oscilatoria)
    vib = 1.0 + np.cos(timestamps * 0.5) * 0.3
    # Presión (estable con pequeñas variaciones)
    press = 2.0 + np.sin(timestamps * 0.05) * 0.1
    
    # 3. Inyectar anomalías / fallas según el tipo
    label = 0 # 0: Normal, 1: Falla
    anomaly_desc = "Operación Normal"
    
    if anomaly_type == 'spikes':
        label = 1
        anomaly_desc = "Falla: Picos transitorios anormales de vibración"
        # Inyectar picos severos de vibración en puntos aleatorios de la segunda mitad de la secuencia
        spike_indices = np.random.choice(range(seq_len // 2, seq_len), size=min(3, seq_len // 4), replace=False)
        vib[spike_indices] += np.random.uniform(3.0, 5.0, size=len(spike_indices))
        
    elif anomaly_type == 'thermal':
        label = 1
        anomaly_desc = "Falla: Sobrecalentamiento progresivo (Fuga Térmica)"
        # Aumento exponencial de temperatura a partir de la mitad de la secuencia
        start_idx = seq_len // 2
        thermal_growth = np.exp(np.linspace(0, 3, seq_len - start_idx)) * 4.0
        temp[start_idx:] += thermal_growth
        # La presión aumenta ligeramente debido a la temperatura
        press[start_idx:] += thermal_growth * 0.05
        
    elif anomaly_type == 'leak':
        label = 1
        anomaly_desc = "Falla: Caída abrupta de presión (Fuga de fluido/gas)"
        # Pérdida repentina de presión en el último tercio
        leak_idx = int(seq_len * 0.7)
        press[leak_idx:] = press[leak_idx:] - np.linspace(0.8, 1.2, seq_len - leak_idx)
        # La vibración puede desestabilizarse
        vib[leak_idx:] += np.random.normal(0, 0.4, size=seq_len - leak_idx)

    # 4. Añadir ruido de medición (ruido blanco gaussiano)
    temp += np.random.normal(0, noise_level * 5.0, size=seq_len)
    vib += np.random.normal(0, noise_level * 0.5, size=seq_len)
    press += np.random.normal(0, noise_level * 0.2, size=seq_len)
    
    # 5. Aplicar pérdida de datos (datos faltantes / NaNs)
    if missing_rate > 0.0:
        # Creamos máscaras booleanas para cada sensor
        temp_mask = np.random.rand(seq_len) < missing_rate
        vib_mask = np.random.rand(seq_len) < missing_rate
        press_mask = np.random.rand(seq_len) < missing_rate
        
        # Colocamos NaNs en los datos faltantes
        temp[temp_mask] = np.nan
        vib[vib_mask] = np.nan
        press[press_mask] = np.nan
        
    # Crear DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'dt': dt_intervals,
        'sensor_temp': temp,
        'sensor_vib': vib,
        'sensor_press': press
    })
    
    return df, label, anomaly_desc


def preprocess_and_impute(df, method='forward_fill'):
    """
    Función de preprocesamiento para el tratamiento de datos faltantes e irregularidades.
    Imputa NaNs y prepara los sensores para el modelo PyTorch.
    
    Métodos de imputación:
      - 'forward_fill': Rellena con el último valor válido observado (ideal para tiempo real).
      - 'mean': Rellena con la media del sensor.
      - 'zero': Rellena con 0.
    """
    df_imputed = df.copy()
    
    # Valores por defecto para inicialización si el primer elemento es NaN
    defaults = {
        'sensor_temp': 70.0,
        'sensor_vib': 1.0,
        'sensor_press': 2.0
    }
    
    for col, default_val in defaults.items():
        if df_imputed[col].isnull().all():
            df_imputed[col] = default_val
        elif method == 'forward_fill':
            # Rellena hacia adelante y luego hacia atrás por si el primero es NaN
            df_imputed[col] = df_imputed[col].ffill().bfill()
        elif method == 'mean':
            mean_val = df_imputed[col].mean()
            if pd.isnull(mean_val):
                mean_val = default_val
            df_imputed[col] = df_imputed[col].fillna(mean_val)
        elif method == 'zero':
            df_imputed[col] = df_imputed[col].fillna(0.0)
            
    return df_imputed


def prepare_tensors_for_model(df_imputed):
    """
    Convierte el DataFrame preprocesado en tensores listos para el modelo PyTorch.
    Devuelve:
      - sensor_tensor: [1, seq_len, 3] (batch size = 1)
      - dt_tensor: [1, seq_len, 1] (intervalo temporal de cada paso)
    """
    # Sensores a normalizar (Escalamiento básico para estabilidad del gradiente)
    temp = (df_imputed['sensor_temp'].values - 70.0) / 10.0
    vib = (df_imputed['sensor_vib'].values - 1.0) / 2.0
    press = (df_imputed['sensor_press'].values - 2.0) / 1.0
    
    # Combinar sensores en formato [seq_len, 3]
    sensors = np.stack([temp, vib, press], axis=-1)
    
    # Delta t observado
    dts = df_imputed['dt'].values.reshape(-1, 1)
    
    # Convertir a Tensores de PyTorch con dimensión de Batch añadida
    sensor_tensor = torch.tensor(sensors, dtype=torch.float32).unsqueeze(0) # [1, seq_len, 3]
    dt_tensor = torch.tensor(dts, dtype=torch.float32).unsqueeze(0) # [1, seq_len, 1]
    
    return sensor_tensor, dt_tensor
