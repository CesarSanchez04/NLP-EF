import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    Bloque de Espacio de Estados Selectivo (S6) adaptado para CPU.
    Implementa la discretización dependiente de la irregularidad del tiempo.
    """
    def __init__(self, d_model, d_state=16, d_inner=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner

        # Proyecciones para parámetros selectivos (B, C, Delta)
        # B y C dependen de la entrada (selectividad temporal)
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_inner, bias=False)
        
        # Proyección para el tamaño de paso Delta (depende de la entrada)
        self.dt_proj = nn.Linear(d_inner, d_model, bias=True)

        # Parámetro A (matriz de transición del estado oculto, estructurada diagonal)
        # Inicializada según la teoría de Mamba/HiPPO (valores negativos para estabilidad)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        
        # Parámetro D (conexión residual/salto directo)
        self.D = nn.Parameter(torch.ones(d_model))

        # Proyecciones de entrada y salida
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Convolución local de soporte antes de la selectividad
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=4,
            padding=3,
            groups=d_inner
        )

    def forward(self, x, delta_t=None):
        """
        x: Tensor de tamaño [batch_size, seq_len, d_model]
        delta_t: Tensor de tamaño [batch_size, seq_len, 1] que representa el tiempo transcurrido irregular.
                 Si es None, se asume uniforme (delta_t = 1.0).
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Proyección de entrada y separación en dos caminos (estilo Gated Unit)
        x_and_res = self.in_proj(x) # [batch_size, seq_len, d_inner * 2]
        x_inner, res = x_and_res.chunk(2, dim=-1) # split en x_inner y residuo
        
        # 2. Convolución local 1D a lo largo del tiempo
        # PyTorch Conv1d requiere [batch, channels, seq_len]
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # recortar padding
        x_conv = x_conv.transpose(1, 2) # regresar a [batch, seq_len, d_inner]
        
        x_conv = F.silu(x_conv) # Activación SiLU (Swish)

        # 3. Obtener parámetros selectivos B, C, Delta
        # x_proj proyecta a (d_state * 2 + d_inner)
        proj_out = self.x_proj(x_conv)
        dt_raw, B_raw, C_raw = torch.split(
            proj_out, 
            [self.d_inner, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Calcular Delta (dt) combinando la entrada y la irregularidad de tiempo física delta_t
        # dt = softplus(dt_proj(dt_raw) + log(delta_t))
        dt = self.dt_proj(dt_raw) # [batch_size, seq_len, d_model]
        if delta_t is not None:
            # Añadimos el logaritmo del delta_t físico observado en el sensor para escalar el paso
            dt = dt + torch.log(delta_t + 1e-5)
        dt = F.softplus(dt) # Garantizar que dt > 0

        # Discretizar la matriz A (A = -exp(A_log))
        A = -torch.exp(self.A_log) # [d_model, d_state]

        # 4. Escaneo Selectivo (Selective Scan) recursivo adaptado a CPU
        # Ecuaciones de espacio de estados discretizadas:
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        # y_t = C * h_t
        
        # Inicializar estado oculto h en ceros: [batch_size, d_model, d_state]
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        y = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)

        for t in range(seq_len):
            # dt_t: [batch, d_model]
            dt_t = dt[:, t, :]
            # B_t: [batch, d_state], C_t: [batch, d_state]
            B_t = B_raw[:, t, :]
            C_t = C_raw[:, t, :]
            # x_t: [batch, d_inner] -> proyectada para emparejar dimensiones ocultas
            # Hacemos una aproximación de broadcast
            x_t = x_conv[:, t, :] # [batch, d_inner]

            # Discretización ZOH (Zero-Order Hold)
            # A_bar = exp(dt * A) -> [batch, d_model, d_state]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            
            # B_bar = dt * B -> [batch, d_model, d_state] via outer product
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # Actualización del estado oculto
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            # x_t.unsqueeze(1) tiene tamaño [batch, 1, d_inner]
            # Usamos aproximación lineal para combinar con el estado
            x_proj_state = x_t.unsqueeze(1).repeat(1, self.d_model, 1) # [batch, d_model, d_inner]
            h = A_bar * h + B_bar * x_proj_state[:, :, :self.d_state]

            # Ecuación de salida discretizada: y_t = C * h_t
            # h: [batch, d_model, d_state], C_t: [batch, d_state]
            # y_t: [batch, d_model]
            y[:, t, :] = torch.sum(h * C_t.unsqueeze(1), dim=-1)

        # 5. Camino residual y multiplicación de compuerta (Gated connection)
        y = y * F.silu(res)
        
        # Proyección de salida
        out = self.out_proj(y)
        
        # Conexión residual directa (parámetro D)
        out = out + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return out


class BiMambaBlock(nn.Module):
    """
    Bloque Mamba Bidireccional (Bi-Mamba).
    Procesa las secuencias de sensores tanto hacia adelante como en reversa
    para capturar contexto temporal completo antes y después de una falla.
    """
    def __init__(self, d_model, d_state=16, d_inner=32):
        super().__init__()
        # Dirección hacia adelante (Forward)
        self.mamba_forward = SelectiveSSM(d_model, d_state, d_inner)
        # Dirección hacia atrás (Backward)
        self.mamba_backward = SelectiveSSM(d_model, d_state, d_inner)
        
        # Proyección de fusión de ambas direcciones
        self.merge_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, delta_t=None):
        """
        x: [batch_size, seq_len, d_model]
        delta_t: [batch_size, seq_len, 1]
        """
        # Procesamiento hacia adelante
        out_forward = self.mamba_forward(x, delta_t)

        # Procesamiento hacia atrás (revertir secuencias)
        x_backward = torch.flip(x, dims=[1])
        delta_t_backward = torch.flip(delta_t, dims=[1]) if delta_t is not None else None
        
        out_backward_raw = self.mamba_backward(x_backward, delta_t_backward)
        # Revertir el resultado para alinearlo con el tiempo original
        out_backward = torch.flip(out_backward_raw, dims=[1])

        # Concatenar y proyectar para fusionar la información bidireccional
        out_merged = torch.cat([out_forward, out_backward], dim=-1)
        out = self.merge_proj(out_merged)
        
        return out


class BiMambaIIoTFaultDetector(nn.Module):
    """
    Red Neuronal Completa basada en Bi-Mamba para Clasificación de Fallas IIoT.
    Toma secuencias de sensores de dimensiones variables y produce diagnóstico.
    """
    def __init__(self, num_sensors, d_model=32, d_state=16, num_classes=2):
        super().__init__()
        # Capa de embedding para los sensores de entrada
        self.sensor_embedding = nn.Linear(num_sensors, d_model)
        
        # Bloques Bi-Mamba secuenciales (2 bloques para profundidad)
        self.bimamba_block1 = BiMambaBlock(d_model, d_state)
        self.bimamba_block2 = BiMambaBlock(d_model, d_state)
        
        # Capa de normalización
        self.norm = nn.LayerNorm(d_model)
        
        # Clasificador final (Falla vs Normal, o multi-clase si se desea)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, sensor_seq, delta_t=None):
        """
        sensor_seq: [batch_size, seq_len, num_sensors]
        delta_t: [batch_size, seq_len, 1] (irregularidades temporales)
        """
        # 1. Proyección inicial a espacio d_model
        x = self.sensor_embedding(sensor_seq)
        
        # 2. Paso por los bloques Bi-Mamba
        x = self.bimamba_block1(x, delta_t)
        x = self.bimamba_block2(x, delta_t)
        x = self.norm(x)
        
        # 3. Pooling temporal (tomamos la representación del último paso
        # o el promedio temporal. Para secuencias temporales, el promedio es más robusto)
        x_pooled = torch.mean(x, dim=1) # [batch_size, d_model]
        
        # 4. Clasificación final
        logits = self.classifier(x_pooled) # [batch_size, num_classes]
        
        return logits
