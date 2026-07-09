import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.utils import prepare_data_loaders
from app.model import LSTMClassifier, GRUClassifier, BiMambaClassifier

# Configuración de estilo visual
sns.set_theme(style="whitegrid")

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=20, device='cpu'):
    print(f"\n================ Entrenamiento de {model_name} ================")
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(1, epochs + 1):
        # Modo Entrenamiento
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seqs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Modo Validación
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * seqs.size(0)
                
                preds = (outputs >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Calcular métricas de validación
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
            
        # Early stopping por pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping activado en época {epoch} tras no mejorar durante {patience} épocas.")
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Evaluación final con el mejor estado
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            preds = (outputs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    return model, history, metrics

def main():
    csv_path = 'data/sensores_iiot_simulados.csv'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device}")
    
    # 1. Preparar cargadores de datos
    train_loader, val_loader, scaler = prepare_data_loaders(
        csv_path=csv_path,
        batch_size=4,
        val_split=0.3,
        method='forward_fill',
        scaler_save_path=os.path.join(results_dir, 'scaler.pkl'),
        random_seed=42
    )
    
    criterion = nn.BCELoss()
    
    # 2. Entrenar LSTM
    lstm_model = LSTMClassifier(input_dim=4, hidden_dim=32, num_layers=2, dropout=0.2)
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.005, weight_decay=1e-4)
    lstm_model, lstm_history, lstm_metrics = train_model(
        'LSTM', lstm_model, train_loader, val_loader, criterion, optimizer_lstm, epochs=100, patience=20, device=device
    )
    torch.save(lstm_model.state_dict(), os.path.join(results_dir, 'lstm_baseline.pth'))
    
    # 3. Entrenar GRU
    gru_model = GRUClassifier(input_dim=4, hidden_dim=32, num_layers=2, dropout=0.2)
    optimizer_gru = optim.Adam(gru_model.parameters(), lr=0.005, weight_decay=1e-4)
    gru_model, gru_history, gru_metrics = train_model(
        'GRU', gru_model, train_loader, val_loader, criterion, optimizer_gru, epochs=100, patience=20, device=device
    )
    torch.save(gru_model.state_dict(), os.path.join(results_dir, 'gru_baseline.pth'))
    
    # 4. Entrenar Bi-Mamba
    bimamba_model = BiMambaClassifier(input_dim=4, d_model=32, d_state=16)
    optimizer_mamba = optim.Adam(bimamba_model.parameters(), lr=0.005, weight_decay=1e-4)
    bimamba_model, mamba_history, mamba_metrics = train_model(
        'Bi-Mamba', bimamba_model, train_loader, val_loader, criterion, optimizer_mamba, epochs=100, patience=20, device=device
    )
    torch.save(bimamba_model.state_dict(), os.path.join(results_dir, 'bimamba_model.pth'))
    
    # 5. Guardar tabla comparativa de métricas
    comparison_data = {
        'Model': ['LSTM', 'GRU', 'Bi-Mamba'],
        'Accuracy': [lstm_metrics['accuracy'], gru_metrics['accuracy'], mamba_metrics['accuracy']],
        'Precision': [lstm_metrics['precision'], gru_metrics['precision'], mamba_metrics['precision']],
        'Recall': [lstm_metrics['recall'], gru_metrics['recall'], mamba_metrics['recall']],
        'F1-Score': [lstm_metrics['f1'], gru_metrics['f1'], mamba_metrics['f1']]
    }
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(os.path.join(results_dir, 'metrics_comparison.csv'), index=False)
    print("\nTabla comparativa de métricas:")
    print(df_comparison)
    
    # 6. Graficar curvas de pérdida comparadas (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # LSTM Curves
    axes[0].plot(lstm_history['train_loss'], label='Entrenamiento', color='#1E3A8A', lw=2)
    axes[0].plot(lstm_history['val_loss'], label='Validación', color='#EF4444', lw=2, linestyle='--')
    axes[0].set_title('Pérdida - LSTM Baseline', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Época', fontsize=10)
    axes[0].set_ylabel('Pérdida (BCELoss)', fontsize=10)
    axes[0].legend(fontsize=9)
    
    # GRU Curves
    axes[1].plot(gru_history['train_loss'], label='Entrenamiento', color='#065F46', lw=2)
    axes[1].plot(gru_history['val_loss'], label='Validación', color='#F59E0B', lw=2, linestyle='--')
    axes[1].set_title('Pérdida - GRU Baseline', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Época', fontsize=10)
    axes[1].set_ylabel('Pérdida (BCELoss)', fontsize=10)
    axes[1].legend(fontsize=9)
    
    # Bi-Mamba Curves
    axes[2].plot(mamba_history['train_loss'], label='Entrenamiento', color='#6D28D9', lw=2)
    axes[2].plot(mamba_history['val_loss'], label='Validación', color='#EC4899', lw=2, linestyle='--')
    axes[2].set_title('Pérdida - Bi-Mamba', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Época', fontsize=10)
    axes[2].set_ylabel('Pérdida (BCELoss)', fontsize=10)
    axes[2].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves_comparison.png'), dpi=300)
    plt.close()
    
    # 7. Graficar matrices de confusión (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels_text = ['Normal', 'Falla']
    
    # LSTM Matrix
    sns.heatmap(lstm_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_text, yticklabels=labels_text, ax=axes[0], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title('Matriz de Confusión - LSTM', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicción', fontsize=10)
    axes[0].set_ylabel('Real', fontsize=10)
    
    # GRU Matrix
    sns.heatmap(gru_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=labels_text, yticklabels=labels_text, ax=axes[1], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title('Matriz de Confusión - GRU', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicción', fontsize=10)
    axes[1].set_ylabel('Real', fontsize=10)
    
    # Bi-Mamba Matrix
    sns.heatmap(mamba_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=labels_text, yticklabels=labels_text, ax=axes[2], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[2].set_title('Matriz de Confusión - Bi-Mamba', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Predicción', fontsize=10)
    axes[2].set_ylabel('Real', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    print("\n¡Entrenamiento y actualización de reportes completados con éxito!")

if __name__ == '__main__':
    main()
