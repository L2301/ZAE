import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionCompression(nn.Module):
    """Single cross-attention layer that compresses sequence length."""
    
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
        queries: (B, target_len, D) - compression tokens
        kv: (B, source_len, D) - input sequence to compress
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


class SequenceEncoder(nn.Module):
    """
    Encoder that uses hierarchical cross-attention to compress variable-length 
    sequences to a single 768-dim vector.
    
    Key properties:
    - Handles variable sequence lengths
    - Actually compresses via learned compression tokens
    - Maintains input variance in output (prevents collapse)
    - Avoids collision with vocabulary embedding vectors
    """
    
    def __init__(self, d_model=768, n_heads=8, n_compression_stages=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_compression_stages = n_compression_stages
        
        # Compression ratios for each stage (e.g., 128 -> 32 -> 8 -> 1)
        self.compression_ratios = [4, 4, 8]  # Customize as needed
        
        # Learnable compression tokens for each stage
        self.compression_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, self._get_stage_tokens(i), d_model))
            for i in range(n_compression_stages)
        ])
        
        # Cross-attention layers for each compression stage
        self.cross_attentions = nn.ModuleList([
            CrossAttentionCompression(d_model, n_heads, dropout)
            for _ in range(n_compression_stages)
        ])
        
        # Feed-forward layers after each compression
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_compression_stages)
        ])
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def _get_stage_tokens(self, stage_idx):
        """Calculate number of compression tokens for each stage."""
        if stage_idx == self.n_compression_stages - 1:
            return 1  # Final stage always outputs 1 token
        # Work backwards from final stage
        tokens = 1
        for i in range(self.n_compression_stages - 1, stage_idx, -1):
            tokens *= self.compression_ratios[i]
        return tokens
    
    def forward(self, x):
        """
        x: (batch_size, seq_length, 768) or (batch_size, 768, seq_length)
        returns: (batch_size, 768)
        """
        # Handle both input formats
        if x.shape[1] == 768 and x.shape[2] != 768:
            x = x.transpose(1, 2)  # (B, seq_len, 768)
        
        B = x.shape[0]
        
        # Hierarchical compression
        compressed = x
        for i, (cross_attn, ffn, tokens) in enumerate(zip(
            self.cross_attentions, self.ffns, self.compression_tokens
        )):
            # Expand compression tokens for batch
            queries = tokens.expand(B, -1, -1)
            
            # Cross-attention compression
            compressed = cross_attn(queries, compressed)
            
            # Feed-forward
            compressed = ffn(compressed) + compressed
        
        # Should be (B, 1, 768) now
        encoded = compressed.squeeze(1)  # (B, 768)
        
        # Final projection
        encoded = self.output_proj(encoded)
        
        return encoded
    
    def compute_variance_loss(self, encoded_batch):
        """
        Regularization to maintain variance in outputs.
        Encourages output variance to match a target.
        """
        batch_var = torch.var(encoded_batch, dim=0).mean()
        target_var = 1.0  # Target variance per dimension
        variance_loss = F.mse_loss(batch_var, torch.tensor(target_var, device=encoded_batch.device))
        return variance_loss
    
    def compute_vocab_repulsion_loss(self, encoded_batch, vocab_embeddings, margin=0.1):
        """
        Repulsion loss to keep encoded vectors away from vocabulary embeddings.
        
        vocab_embeddings: (vocab_size, d_model) - the embedding matrix from GPT
        margin: minimum desired distance from vocab vectors
        """
        # Normalize vectors
        encoded_norm = F.normalize(encoded_batch, dim=-1)  # (B, D)
        vocab_norm = F.normalize(vocab_embeddings, dim=-1)  # (V, D)
        
        # Compute similarities to all vocab vectors
        similarities = encoded_norm @ vocab_norm.T  # (B, V)
        
        # Find max similarity for each encoded vector
        max_similarities = similarities.max(dim=-1)[0]  # (B,)
        
        # Penalize high similarities (want them below 1-margin)
        target_similarity = 1.0 - margin
        repulsion_loss = F.relu(max_similarities - target_similarity).mean()
        
        return repulsion_loss
    
    def compute_contrastive_loss(self, input_encoded, hidden_encoded, temperature=0.07):
        """
        Contrastive loss to ensure input sequences and their corresponding 
        GPT hidden states map to similar vectors.
        
        input_encoded: (B, D) - encoded input sequences
        hidden_encoded: (B, D) - encoded hidden state sequences
        """
        # Normalize
        input_norm = F.normalize(input_encoded, dim=-1)
        hidden_norm = F.normalize(hidden_encoded, dim=-1)
        
        # Similarity matrix
        similarity = (input_norm @ hidden_norm.T) / temperature  # (B, B)
        
        # Labels: diagonal entries are positive pairs
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        
        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(similarity, labels)
        
        return loss