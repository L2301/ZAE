"""
Sequence decoder - expands single vector back to sequence
"""

import torch
import torch.nn as nn


class SequenceDecoder(nn.Module):
    """
    Decoder: 768 -> bottleneck -> (seq_length, 768)
    Expands a single vector into a sequence of vectors
    """
    
    def __init__(self, seq_length, hidden_dim=768, bottleneck_dim=256):
        super().__init__()
        
        # Compress input
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2)
        )
        
        # Expand to sequence
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            nn.GELU()
        )
        
        # Per-position output projection
        self.post_expand = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Learnable position queries
        self.position_queries = nn.Parameter(torch.randn(seq_length, bottleneck_dim))
        
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
        # Expand bottleneck
        expanded = self.decoder(z)  # (batch, bottleneck_dim)
        
        # Add position queries
        batch_size = z.size(0)
        pos_queries = self.position_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_length, bottleneck_dim)
        combined = expanded.unsqueeze(1) + pos_queries  # (batch, seq_length, bottleneck_dim)
        
        # Project to output dimension
        output = self.post_expand(combined)  # (batch, seq_length, 768)
        return output
    
    def forward(self, x):
        """
        Args:
            x: (batch, 768) single vector
        Returns:
            (batch, seq_length, 768) reconstructed sequence
        """
        z = self.encode(x)
        return self.decode(z)