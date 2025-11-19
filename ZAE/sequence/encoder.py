"""
Flat sequence autoencoder - compresses flattened sequence into one vector
High parameter count version
"""

import torch
import torch.nn as nn


class FlatSequenceEncoder(nn.Module):
    """
    Encoder: (seq_length * 768) -> bottleneck -> 768
    Flattens and compresses a sequence of vectors into a single vector
    """
    
    def __init__(self, seq_length, hidden_dim=768, bottleneck_dim=256):
        super().__init__()
        
        input_dim = seq_length * hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, hidden_dim)
        )
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
    
    def encode(self, x):
        """
        Args:
            x: (batch, seq_length, 768) sequence of vectors
        Returns:
            (batch, bottleneck_dim) compressed representation
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # (batch, seq_length * 768)
        return self.encoder(x_flat)
    
    def decode(self, z):
        """
        Args:
            z: (batch, bottleneck_dim) compressed vectors
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