"""
Gumbel-Softmax classifier for discrete decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelClassifier(nn.Module):
    """
    Learnable Gumbel-Softmax classifier that maps a vector to {0, 1, 2}
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x, tau=1.0, hard=False):
        """
        Args:
            x: (batch, input_dim) input vectors
            tau: temperature for Gumbel-Softmax (lower = more discrete)
            hard: if True, returns one-hot; if False, returns soft probabilities
        Returns:
            if hard: (batch, num_classes) one-hot vectors
            if not hard: (batch, num_classes) soft probabilities
        """
        logits = self.fc2(F.relu(self.fc1(x)))
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        return y
    
    def classify(self, x):
        """
        Deterministic classification (argmax)
        
        Args:
            x: (batch, input_dim) input vectors
        Returns:
            (batch,) class indices in {0, 1, 2}
        """
        logits = self.fc2(F.relu(self.fc1(x)))
        return torch.argmax(logits, dim=-1)