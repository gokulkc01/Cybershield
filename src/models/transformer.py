"""
transformer.py
==============
PyTorch Transformer Encoder architecture for CyberShield.

Ingests unflattened (Batch, 20, 12) sequence tensors.
Uses a CLS token approach (like BERT) to aggregate temporal features
and output a binary C2 classification logit.
"""

import torch
import torch.nn as nn

class C2Transformer(nn.Module):
    def __init__(
        self, 
        feature_dim: int = 12, 
        seq_len: int = 20, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 3, 
        dim_feedforward: int = 128, 
        dropout: float = 0.2
    ):
        super(C2Transformer, self).__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 1. Input Projection & Stabilization
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # 2. Learnable CLS Token & Positional Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Stabilization (NEW FIX)
        self.final_norm = nn.LayerNorm(d_model)
        
        # 5. Classification Head (Kept Shallow to prevent overfitting)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Logit output
        )
        
        self._init_weights()

    def _init_weights(self):
        """Standardized initialization for transformer stability."""
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'pos_encoder' not in n and 'cls_token' not in n:
                nn.init.xavier_uniform_(p)
        
        nn.init.normal_(self.pos_encoder, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        
        # Project and stabilize inputs
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Masking
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        
        # Positional Encoding
        x = x + self.pos_encoder
        
        # Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Final Norm (NEW FIX)
        x = self.final_norm(x)
        
        # Extract CLS token
        cls_out = x[:, 0, :]
        
        # Classifier
        logits = self.classifier(cls_out)
        
        return logits.squeeze(-1)