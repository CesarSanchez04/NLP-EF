import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Clasificador basado en LSTM para secuencias de telemetría de sensores IIoT.
        
        Parámetros:
          - input_dim: Dimensión de entrada de la secuencia (ej. 4 características).
          - hidden_dim: Dimensión del estado oculto de la LSTM.
          - num_layers: Número de capas LSTM apiladas.
          - dropout: Tasa de dropout para regularización (solo si num_layers > 1).
        """
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Entrada x: (batch_size, seq_len, input_dim)
        out = self.fc_in(x)  # Proyección lineal: (batch_size, seq_len, hidden_dim)
        out, (hn, cn) = self.lstm(out)  # out: (batch_size, seq_len, hidden_dim)
        
        # Global Average Pooling temporal a lo largo de la secuencia (dim=1)
        # Esto mitiga el desvanecimiento del gradiente e imposibilidad de recordar 
        # a largo plazo en secuencias muy extensas (5000 pasos).
        out = torch.mean(out, dim=1)  # (batch_size, hidden_dim)
        
        logits = self.fc_out(out)  # (batch_size, 1)
        probs = self.sigmoid(logits)  # Probabilidad binaria de anomalía (0 o 1)
        return probs

class GRUClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Clasificador basado en GRU para secuencias de telemetría de sensores IIoT.
        
        Parámetros:
          - input_dim: Dimensión de entrada de la secuencia (ej. 4 características).
          - hidden_dim: Dimensión del estado oculto de la GRU.
          - num_layers: Número de capas GRU apiladas.
          - dropout: Tasa de dropout para regularización (solo si num_layers > 1).
        """
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Entrada x: (batch_size, seq_len, input_dim)
        out = self.fc_in(x)  # Proyección lineal: (batch_size, seq_len, hidden_dim)
        out, hn = self.gru(out)  # out: (batch_size, seq_len, hidden_dim)
        
        # Global Average Pooling temporal a lo largo de la secuencia (dim=1)
        out = torch.mean(out, dim=1)  # (batch_size, hidden_dim)
        
        logits = self.fc_out(out)  # (batch_size, 1)
        probs = self.sigmoid(logits)  # Probabilidad binaria de anomalía (0 o 1)
        return probs
