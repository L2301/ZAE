"""
Sequence autoencoder - compresses multiple vectors into one
"""

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """
    Encoder: (seq_length, 768) -> bottleneck -> 768
    Compresses a sequence of vectors into a single vector using learned pooling
    """
    
    def __init__(self, seq_length, hidden_dim=768, bottleneck_dim=256):
        super().__init__()
        
        # Per-vector projection before pooling
        self.pre_pool = nn.Linear(hidden_dim, bottleneck_dim)
        
        # Learned attention weights for pooling
        self.pool_weights = nn.Linear(bottleneck_dim, 1)
        
        # Bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.GELU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim)
        )
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
    
    def encode(self, x):
        """
        Args:
            x: (batch, seq_length, 768) sequence of vectors
        Returns:
            (batch, bottleneck_dim//2) compressed representation
        """
        # Project each vector
        x_proj = self.pre_pool(x)  # (batch, seq_length, bottleneck_dim)
        
        # Learned attention pooling
        attn_logits = self.pool_weights(x_proj)  # (batch, seq_length, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch, seq_length, 1)
        pooled = (x_proj * attn_weights).sum(dim=1)  # (batch, bottleneck_dim)
        
        # Further compression
        z = self.encoder(pooled)  # (batch, bottleneck_dim//2)
        return z
    
    def decode(self, z):
        """
        Args:
            z: (batch, bottleneck_dim//2) compressed vectors
        Returns:
            (batch, 768) reconstructed single vector
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, 768) sequence of vectors
        Returns:
            (batch, 768) single reconstructed vector
        """
        z = self.encode(x)
        return self.decode(z)