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


class MambaSSMLayer(nn.Module):
    def __init__(self, d_model=32, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Parámetro A (inicializado positivo para que exp(-delta * A) sea estable)
        self.A_param = nn.Parameter(torch.log(torch.arange(1, d_model + 1).float().unsqueeze(1).expand(d_model, d_state)))
        
        # Proyecciones para obtener delta, B y C
        self.x_proj = nn.Linear(d_model, d_state + d_state + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)
        
        # Inicializaciones
        nn.init.normal_(self.A_param, mean=0.5, std=0.1)

    def forward(self, x, dt):
        # x: (batch_size, seq_len, d_model)
        # dt: (batch_size, seq_len)
        batch_size, seq_len, _ = x.shape
        
        # Obtener A
        A = -torch.exp(self.A_param)  # (d_model, d_state)
        
        # Proyectar para obtener delta, B, C
        proj = self.x_proj(x)  # (batch_size, seq_len, d_state * 2 + d_model)
        B, C, dt_step = torch.split(proj, [self.d_state, self.d_state, self.d_model], dim=-1)
        
        # Calcular delta multiplicando por el dt físico (pasado por softplus para asegurar positividad y estabilidad)
        dt_positive = torch.nn.functional.softplus(dt)
        delta = torch.sigmoid(self.dt_proj(dt_step)) * dt_positive.unsqueeze(-1)  # (batch_size, seq_len, d_model)
        
        # Inicializar estado oculto
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        ys = []
        
        # Recurrencia selectiva
        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)  # (batch_size, d_model, 1)
            B_t = B[:, t, :].unsqueeze(1)          # (batch_size, 1, d_state)
            C_t = C[:, t, :].unsqueeze(-1)         # (batch_size, d_state, 1)
            x_t = x[:, t, :].unsqueeze(-1)         # (batch_size, d_model, 1)
            
            # Discretización
            A_bar = torch.exp(delta_t * A.unsqueeze(0))  # (batch_size, d_model, d_state)
            B_bar = delta_t * B_t  # (batch_size, d_model, d_state)
            
            # Actualización de estado
            h = A_bar * h + B_bar * x_t.expand(-1, -1, self.d_state)
            
            # Salida y_t: (batch_size, d_model)
            y_t = torch.matmul(h, C_t).squeeze(-1)
            ys.append(y_t)
            
        return torch.stack(ys, dim=1)  # (batch_size, seq_len, d_model)


class BiMambaClassifier(nn.Module):
    def __init__(self, input_dim=4, d_model=32, d_state=16):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        
        # Bloques SSM en ambas direcciones
        self.mamba_fwd = MambaSSMLayer(d_model=d_model, d_state=d_state)
        self.mamba_bwd = MambaSSMLayer(d_model=d_model, d_state=d_state)
        
        self.fc_out = nn.Linear(2 * d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # Extraemos el dt que es la primera característica (columna 0)
        dt = x[:, :, 0]
        
        # Entrada proyectada
        out = self.fc_in(x)  # (batch_size, seq_len, d_model)
        
        # Rama Forward
        fwd_out = self.mamba_fwd(out, dt)
        
        # Rama Backward (revertir la secuencia en dimensión temporal dim=1)
        x_reversed = torch.flip(out, dims=[1])
        dt_reversed = torch.flip(dt, dims=[1])
        bwd_out_reversed = self.mamba_bwd(x_reversed, dt_reversed)
        bwd_out = torch.flip(bwd_out_reversed, dims=[1])
        
        # Fusión
        merged = torch.cat([fwd_out, bwd_out], dim=-1)  # (batch_size, seq_len, 2 * d_model)
        
        # Pooling temporal
        pooled = torch.mean(merged, dim=1)  # (batch_size, 2 * d_model)
        
        # Clasificación
        logits = self.fc_out(pooled)
        probs = self.sigmoid(logits)
        return probs

