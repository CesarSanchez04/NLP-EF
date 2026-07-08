import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def preprocess_and_impute(df, method='forward_fill'):
    """
    Imputa NaNs en las columnas de sensores del DataFrame.
    
    Métodos:
      - 'forward_fill': Copia el último valor válido hacia adelante (ffill) y hacia atrás (bfill).
      - 'mean': Reemplaza con la media de la secuencia.
      - 'zero': Reemplaza con cero.
    """
    df_imputed = df.copy()
    
    defaults = {
        'sensor_temp': 70.0,
        'sensor_vib': 1.0,
        'sensor_press': 2.0
    }
    
    for col, default_val in defaults.items():
        if df_imputed[col].isnull().all():
            df_imputed[col] = default_val
        elif method == 'forward_fill':
            df_imputed[col] = df_imputed[col].ffill().bfill()
        elif method == 'mean':
            mean_val = df_imputed[col].mean()
            if pd.isnull(mean_val):
                mean_val = default_val
            df_imputed[col] = df_imputed[col].fillna(mean_val)
        elif method == 'zero':
            df_imputed[col] = df_imputed[col].fillna(0.0)
            
    return df_imputed

class IIoTDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Dataset personalizado para secuencias de sensores IIoT.
        
        Parámetros:
          - sequences: Lista de matrices numpy, cada una de forma (seq_len, num_features).
          - labels: Lista de etiquetas (0 para normal, 1 para falla).
        """
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N, 1) para BCELoss

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def prepare_data_loaders(csv_path, batch_size=4, val_split=0.3, method='forward_fill', scaler_save_path='results/scaler.pkl', random_seed=42):
    """
    Carga el dataset CSV, realiza la división a nivel de secuencia, imputa valores faltantes,
    normaliza las características con StandardScaler y crea DataLoaders de PyTorch.
    
    Retorna:
      - train_loader, val_loader, scaler
    """
    # 1. Cargar el DataFrame completo
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV en {csv_path}. Ejecuta primero el cuaderno de simulación.")
        
    df = pd.read_csv(csv_path)
    
    # 2. Dividir secuencias de manera que las de prueba/validación no se hayan visto en entrenamiento
    unique_ids = df['sequence_id'].unique()
    
    np.random.seed(random_seed)
    np.random.shuffle(unique_ids)
    
    # Si val_split=0.3, train tiene 70%
    split_idx = int(len(unique_ids) * (1.0 - val_split))
    
    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]
    
    print(f"IDs de secuencia para Entrenamiento: {train_ids}")
    print(f"IDs de secuencia para Validación: {val_ids}")
    
    feature_cols = ['dt', 'sensor_temp', 'sensor_vib', 'sensor_press']
    
    # 3. Procesar secuencias por separado (imputación individual para evitar fugas temporales)
    train_seqs_raw = []
    train_labels = []
    for seq_id in train_ids:
        seq_df = df[df['sequence_id'] == seq_id].sort_values('timestamp')
        seq_imputed = preprocess_and_impute(seq_df, method=method)
        train_seqs_raw.append(seq_imputed[feature_cols].values)
        # La etiqueta es constante en toda la secuencia
        train_labels.append(seq_df['label'].iloc[0])
        
    val_seqs_raw = []
    val_labels = []
    for seq_id in val_ids:
        seq_df = df[df['sequence_id'] == seq_id].sort_values('timestamp')
        seq_imputed = preprocess_and_impute(seq_df, method=method)
        val_seqs_raw.append(seq_imputed[feature_cols].values)
        val_labels.append(seq_df['label'].iloc[0])
        
    # 4. Ajustar el escalador solo con los datos de entrenamiento
    # Concatenamos todas las lecturas de entrenamiento para ajustar el StandardScaler
    all_train_readings = np.concatenate(train_seqs_raw, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train_readings)
    
    # Guardar el escalador entrenado para inferencia futura
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Escalador guardado en {scaler_save_path}")
        
    # 5. Transformar (escalar) las secuencias de entrenamiento y validación
    train_seqs = [scaler.transform(seq) for seq in train_seqs_raw]
    val_seqs = [scaler.transform(seq) for seq in val_seqs_raw]
    
    # 6. Crear instancias de Dataset y DataLoader
    train_dataset = IIoTDataset(train_seqs, train_labels)
    val_dataset = IIoTDataset(val_seqs, val_labels)
    
    # Usamos batch_size adecuado (por ejemplo, 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler
