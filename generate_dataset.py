import os
import numpy as np
import pandas as pd

def generate_iiot_sequence(seq_len, anomaly_type, noise_level=0.05, missing_rate=0.15):
    # 1. Intervalos de tiempo irregulares (Delta t)
    dt_intervals = np.random.exponential(scale=0.8, size=seq_len) + 0.2
    timestamps = np.cumsum(dt_intervals)
    
    # 2. Generar parámetros base aleatorios para esta secuencia
    base_temp = np.random.uniform(65.0, 75.0)
    temp_amp = np.random.uniform(1.0, 3.0)
    temp_freq = np.random.uniform(0.05, 0.2)
    
    base_vib = np.random.uniform(0.8, 1.2)
    vib_amp = np.random.uniform(0.1, 0.4)
    vib_freq = np.random.uniform(0.3, 0.7)
    
    base_press = np.random.uniform(1.8, 2.2)
    press_amp = np.random.uniform(0.05, 0.15)
    press_freq = np.random.uniform(0.02, 0.08)
    
    # Señales base
    temp = base_temp + np.sin(timestamps * temp_freq) * temp_amp
    vib = base_vib + np.cos(timestamps * vib_freq) * vib_amp
    press = base_press + np.sin(timestamps * press_freq) * press_amp
    
    label = 0
    anomaly_desc = "Operación Normal"
    
    # 3. Inyección de anomalías con inicio aleatorio
    if anomaly_type is not None:
        label = 1
        start_idx = np.random.randint(int(seq_len * 0.3), int(seq_len * 0.7))
        
        if anomaly_type == 'spikes':
            anomaly_desc = "Falla: Picos transitorios de vibración mecánica"
            num_spikes = np.random.randint(2, 6)
            spike_indices = np.random.choice(range(start_idx, seq_len), size=num_spikes, replace=False)
            vib[spike_indices] += np.random.uniform(3.0, 6.0, size=num_spikes)
            
        elif anomaly_type == 'thermal':
            anomaly_desc = "Falla: Sobrecalentamiento gradual (Fuga térmica)"
            thermal_rate = np.random.uniform(2.5, 3.5)
            thermal_growth = np.exp(np.linspace(0, thermal_rate, seq_len - start_idx)) * np.random.uniform(2.5, 4.5)
            temp[start_idx:] += thermal_growth
            press[start_idx:] += thermal_growth * np.random.uniform(0.03, 0.07)
            
        elif anomaly_type == 'leak':
            anomaly_desc = "Falla: Caída severa de presión (Fuga de fluido)"
            drop_magnitude = np.random.uniform(0.8, 1.4)
            press[start_idx:] -= np.linspace(0, drop_magnitude, seq_len - start_idx)
            vib[start_idx:] += np.random.normal(0, np.random.uniform(0.2, 0.5), size=seq_len - start_idx)

    # 4. Añadir ruido de medición
    temp += np.random.normal(0, noise_level * 5.0, size=seq_len)
    vib += np.random.normal(0, noise_level * 0.5, size=seq_len)
    press += np.random.normal(0, noise_level * 0.2, size=seq_len)
    
    # 5. Aplicar pérdida de datos (NaNs)
    if missing_rate > 0.0:
        temp[np.random.rand(seq_len) < missing_rate] = np.nan
        vib[np.random.rand(seq_len) < missing_rate] = np.nan
        press[np.random.rand(seq_len) < missing_rate] = np.nan
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'dt': dt_intervals,
        'sensor_temp': temp,
        'sensor_vib': vib,
        'sensor_press': press
    })
    return df, label, anomaly_desc

def main():
    os.makedirs('data', exist_ok=True)
    
    seq_len = 500
    lote_df = []
    
    # Definir los tipos de secuencia a generar
    normal_count = 100
    spikes_count = 33
    thermal_count = 33
    leak_count = 34
    
    generators = (
        [(None, normal_count),
         ('spikes', spikes_count),
         ('thermal', thermal_count),
         ('leak', leak_count)]
    )
    
    seq_id = 0
    np.random.seed(42)  # Para reproducibilidad en la generación
    
    for anomaly_type, count in generators:
        for _ in range(count):
            df, label, desc = generate_iiot_sequence(seq_len, anomaly_type, noise_level=0.05, missing_rate=0.15)
            df['sequence_id'] = seq_id
            df['label'] = label
            df['description'] = desc
            lote_df.append(df)
            seq_id += 1
            
    # Concatenar y guardar como csv
    dataset_full = pd.concat(lote_df, ignore_index=True)
    csv_path = 'data/sensores_iiot_simulados.csv'
    dataset_full.to_csv(csv_path, index=False)
    print(f"Dataset generado con éxito. Registros guardados: {len(dataset_full)} en '{csv_path}'")
    print(f"Total secuencias: {seq_id} (Normales: {normal_count}, Spikes: {spikes_count}, Thermal: {thermal_count}, Leak: {leak_count})")

if __name__ == '__main__':
    main()
