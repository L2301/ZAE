"""
Vector autoDecoder - compresses and reconstructs 768-dim vectors
"""

import torch
import torch.nn as nn


class VectorDecoder(nn.Module):
    """
    Decoder: 768 -> bottleneck -> 768
    """
    
    def __init__(self, input_dim=768, bottleneck_dim=128):
        super().__init__()
        assert bottleneck_dim in [128, 192], "bottleneck_dim must be 128 or 192"
        
        self.Decoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, input_dim)
        )
        
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
    
    def encode(self, x):
        """
        Args:
            x: (..., 768) input vectors
        Returns:
            (..., bottleneck_dim) compressed representation
        """
        return self.Decoder(x)
    
    def decode(self, z):
        """
        Args:
            z: (..., bottleneck_dim) compressed vectors
        Returns:
            (..., 768) reconstructed vectors
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Args:
            x: (..., 768) input vectors
        Returns:
            (..., 768) reconstructed vectors
        """
        z = self.encode(x)
        return self.decode(z)