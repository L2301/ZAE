"""
Flat sequence decoder - expands single vector back to flattened sequence
"""

import torch
import torch.nn as nn


class FlatSequenceDecoder(nn.Module):
    """
    Decoder: 768 -> bottleneck -> (seq_length * 768)
    Expands a single vector into a flattened sequence, then reshapes
    """
    
    def __init__(self, seq_length, hidden_dim=768, bottleneck_dim=256):
        super().__init__()
        
        output_dim = seq_length * hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim, output_dim)
        )
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
    
    def encode(self, x):
        """
        Args:
            x: (batch, 768) single vector
        Returns:
            (batch, bottleneck_dim//2) compressed representation
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Args:
            z: (batch, bottleneck_dim//2) compressed vectors
        Returns:
            (batch, seq_length, 768) reconstructed sequence
        """
        x_flat = self.decoder(z)  # (batch, seq_length * 768)
        return x_flat.view(-1, self.seq_length, self.hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 768) single vector
        Returns:
            (batch, seq_length, 768) reconstructed sequence
        """
        z = self.encode(x)
        return self.decode(z)