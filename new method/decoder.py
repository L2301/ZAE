import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionExpansion(nn.Module):
    """Single cross-attention layer that expands sequence length."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, kv):
        """
        queries: (B, target_len, D) - expansion tokens
        kv: (B, source_len, D) - compressed representation to expand from
        returns: (B, target_len, D)
        """
        B, target_len, _ = queries.shape
        source_len = kv.shape[1]
        
        # Project
        Q = self.q_proj(queries).view(B, target_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, source_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, source_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ V).transpose(1, 2).contiguous().view(B, target_len, self.d_model)
        out = self.out_proj(out)
        
        # Residual + norm
        out = self.norm(queries + out)
        
        return out


class SequenceDecoder(nn.Module):
    """
    Decoder that uses hierarchical cross-attention to expand a single 768-dim 
    vector to a variable-length sequence.
    
    Inverse of SequenceEncoder.
    """
    
    def __init__(self, d_model=768, n_heads=8, n_expansion_stages=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_expansion_stages = n_expansion_stages
        
        # Expansion ratios for each stage (e.g., 1 -> 8 -> 32 -> 128)
        self.expansion_ratios = [8, 4, 4]  # Inverse of encoder
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Learnable expansion tokens for each stage
        self.expansion_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, self._get_stage_tokens(i), d_model))
            for i in range(n_expansion_stages)
        ])
        
        # Cross-attention layers for each expansion stage
        self.cross_attentions = nn.ModuleList([
            CrossAttentionExpansion(d_model, n_heads, dropout)
            for _ in range(n_expansion_stages)
        ])
        
        # Feed-forward layers after each expansion
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_expansion_stages)
        ])
        
    def _get_stage_tokens(self, stage_idx):
        """Calculate number of expansion tokens for each stage."""
        tokens = 1
        for i in range(stage_idx + 1):
            tokens *= self.expansion_ratios[i]
        return tokens
    
    def forward(self, x, target_seq_len=None):
        """
        x: (batch_size, 768) - single compressed vector
        target_seq_len: optional, for variable length output
        returns: (batch_size, seq_length, 768)
        """
        B = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Add sequence dimension
        expanded = x.unsqueeze(1)  # (B, 1, 768)
        
        # Hierarchical expansion
        for i, (cross_attn, ffn, tokens) in enumerate(zip(
            self.cross_attentions, self.ffns, self.expansion_tokens
        )):
            # Expand expansion tokens for batch
            queries = tokens.expand(B, -1, -1)
            
            # Cross-attention expansion
            expanded = cross_attn(queries, expanded)
            
            # Feed-forward
            expanded = ffn(expanded) + expanded
        
        # If target length specified and different, interpolate
        if target_seq_len is not None and expanded.shape[1] != target_seq_len:
            # Transpose for interpolation: (B, seq, D) -> (B, D, seq)
            expanded = expanded.transpose(1, 2)
            expanded = F.interpolate(expanded, size=target_seq_len, mode='linear', align_corners=False)
            expanded = expanded.transpose(1, 2)  # Back to (B, seq, D)
        
        return expanded
    
    def compute_reconstruction_loss(self, reconstructed, original):
        """
        MSE loss between reconstructed and original sequences.
        
        reconstructed: (B, seq_len, D)
        original: (B, seq_len, D)
        """
        return F.mse_loss(reconstructed, original)
    
    def compute_smoothness_loss(self, reconstructed):
        """
        Regularization to encourage smooth transitions between adjacent vectors.
        """
        # Compute differences between adjacent vectors
        diffs = reconstructed[:, 1:, :] - reconstructed[:, :-1, :]
        smoothness_loss = (diffs ** 2).mean()
        return smoothness_loss