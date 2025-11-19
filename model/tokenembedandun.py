"""
Embedding and Unembedding matrices for GPT-2
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Token and position embeddings"""
    
    def __init__(self, vocab_size=50257, n_embd=768, block_size=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)  # token embeddings
        self.wpe = nn.Embedding(block_size, n_embd)  # position embeddings
        self.block_size = block_size
    
    def forward(self, idx):
        """
        Args:
            idx: (batch, seq_len) token indices
        Returns:
            (batch, seq_len, n_embd) embeddings
        """
        b, t = idx.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        return tok_emb + pos_emb
    
    def token_only(self, idx):
        """
        Get token embeddings without position information
        
        Args:
            idx: (batch, seq_len) token indices
        Returns:
            (batch, seq_len, n_embd) token embeddings only
        """
        return self.wte(idx)
    
    def add_positions(self, x):
        """
        Add position embeddings to arbitrary sequence of embeddings
        
        Args:
            x: (batch, seq_len, n_embd) embeddings
        Returns:
            (batch, seq_len, n_embd) embeddings with positions added
        """
        b, t, n_embd = x.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        return x + pos_emb


class Unembedding(nn.Module):
    """Project from embedding space back to vocabulary logits"""
    
    def __init__(self, vocab_size=50257, n_embd=768):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_embd) features
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        return self.lm_head(x)