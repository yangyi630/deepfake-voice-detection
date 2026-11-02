import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """
    Conformer block combining Transformer and CNN.
    Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", INTERSPEECH 2020
    
    Architecture:
        1. Feed-forward module (Macaron style)
        2. Multi-head self-attention module
        3. Convolution module
        4. Feed-forward module
        5. Final layer norm
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, kernel_size=31):
        super(ConformerBlock, self).__init__()
        
        # Feed-forward module 1 (Macaron style)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(d_model)
    
    def forward(self, x, src_mask=None):
        # x shape: (batch, seq_len, d_model)
        
        # Feed-forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = residual + self.dropout1(x_attn)
        
        # Convolution module (need to transpose for conv1d)
        residual = x
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.conv_module(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = residual + x
        
        # Feed-forward 2
        x = x + 0.5 * self.ff2(x)
        
        # Final norm
        x = self.norm_final(x)
        
        return x


class ConformerModel(nn.Module):
    """
    Conformer-based model for voice anti-spoofing.
    
    Key advantages for deepfake voice detection:
    1. Captures long-range temporal dependencies via self-attention
    2. Extracts local spectral features via depth-wise convolutions
    3. Macaron-style feed-forward layers improve model capacity
    
    This model is particularly effective at detecting subtle artifacts
    in AI-synthesized speech that span multiple time scales.
    """
    
    def __init__(self, num_classes=2, n_mels=80, d_model=144, 
                 nhead=4, num_layers=6, dim_feedforward=576, dropout=0.1):
        super(ConformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Conformer encoder layers
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        batch_size = x.size(0)
        
        # Reshape input: (batch, 1, n_mels, time) -> (batch, time, n_mels)
        x = x.squeeze(1).transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding (expects sequence first)
        x = x.transpose(0, 1)  # (time, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, time, d_model)
        
        # Conformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Classification
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.classifier(x)
        
        return x
