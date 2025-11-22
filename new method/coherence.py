"""
Single-Head Coherence Attention - Ensures context sharing across sequence boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoherenceAttention(nn.Module):
    """
    Single-head attention layer for cross-sequence coherence.
    
    Takes decoded sequence embeddings and allows information flow across
    sequence boundaries to maintain coherence.
    
    Args:
        d_model: Dimension of embeddings (768 for GPT-2)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model=768, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Square projection matrices (d_model x d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer norm and dropout
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) - decoded sequence embeddings
            mask: Optional (batch, seq_len, seq_len) attention mask
        
        Returns:
            (batch, seq_len, d_model) - coherence-adjusted embeddings
        """
        batch_size, seq_len, _ = x.shape
        
        # Store residual
        residual = x
        
        # Layer norm
        x = self.ln(x)
        
        # Project to Q, K, V (all d_model dimensional)
        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)  # (B, T, d_model)
        V = self.v_proj(x)  # (B, T, d_model)
        
        # Compute attention scores
        # (B, T, d_model) @ (B, d_model, T) -> (B, T, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # (batch, seq_len)
                mask = mask.unsqueeze(1)  # (B, 1, T)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (B, T, T) @ (B, T, d_model) -> (B, T, d_model)
        attn_output = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection
        output = residual + output
        
        return output


def create_sequence_boundary_mask(batch_size, total_seq_len, chunk_size=4):
    """
    Create attention mask that allows attention within chunks and to adjacent chunks.
    
    Args:
        batch_size: Batch size
        total_seq_len: Total sequence length (e.g., 16 for 4 chunks of 4 tokens)
        chunk_size: Size of each decoded sequence chunk (default: 4)
    
    Returns:
        (batch, total_seq_len, total_seq_len) mask where 1 = attend, 0 = mask
    """
    n_chunks = total_seq_len // chunk_size
    mask = torch.zeros(total_seq_len, total_seq_len)
    
    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = (i + 1) * chunk_size
        
        # Allow attention within chunk
        mask[start_i:end_i, start_i:end_i] = 1
        
        # Allow attention to previous chunk (for context)
        if i > 0:
            start_prev = (i - 1) * chunk_size
            end_prev = i * chunk_size
            mask[start_i:end_i, start_prev:end_prev] = 1
        
        # Allow attention to next chunk (for coherence)
        if i < n_chunks - 1:
            start_next = (i + 1) * chunk_size
            end_next = (i + 2) * chunk_size
            mask[start_i:end_i, start_next:end_next] = 1
    
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


if __name__ == '__main__':
    # Test the single-head coherence attention
    batch_size = 2
    seq_len = 16  # 4 chunks of 4 tokens each
    d_model = 768
    
    # Create dummy decoded embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create single-head coherence attention
    coherence_attn = CoherenceAttention(d_model=d_model, dropout=0.1)
    
    # Create boundary mask (optional - allows limited cross-chunk attention)
    mask = create_sequence_boundary_mask(batch_size, seq_len, chunk_size=4)
    
    # Without mask - full attention across all positions
    print("Testing without mask...")
    output_full = coherence_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_full.shape}")
    
    # With mask - constrained attention at boundaries
    print("\nTesting with boundary mask...")
    output_masked = coherence_attn(x, mask=mask)
    print(f"Output shape: {output_masked.shape}")
    
    # Check that outputs are different
    print(f"\nOutputs are different: {not torch.allclose(output_full, output_masked)}")
    
    # Count parameters
    n_params = sum(p.numel() for p in coherence_attn.parameters())
    print(f"\nCoherence attention parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Breakdown
    print("\nParameter breakdown:")
    print(f"  Q projection: {d_model * d_model:,}")
    print(f"  K projection: {d_model * d_model:,}")
    print(f"  V projection: {d_model * d_model:,}")
    print(f"  Out projection: {d_model * d_model:,}")
    print(f"  LayerNorm: {2 * d_model:,}")