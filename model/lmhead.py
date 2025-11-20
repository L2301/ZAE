import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Language modeling head that projects hidden states to vocabulary logits.
    
    Standard implementation from GPT-2: linear projection from d_model to vocab_size.
    Typically shares weights with token embedding matrix (weight tying).
    """
    
    def __init__(self, d_model=768, vocab_size=50257, bias=False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) or (batch_size, d_model)
        Returns:
            logits: (batch_size, seq_len, vocab_size) or (batch_size, vocab_size)
        """
        return self.lm_head(x)
    
    def tie_weights(self, embedding_weight):
        """Tie LM head weights with token embedding matrix.
        
        Args:
            embedding_weight: (vocab_size, d_model) token embedding weight matrix
        """
        # embedding_weight is (vocab_size, d_model)
        # Linear layer weight should be (vocab_size, d_model) for out_features x in_features
        self.lm_head.weight = nn.Parameter(embedding_weight)
    
    @classmethod
    def from_pretrained_gpt2(cls, model_name='gpt2'):
        """Load LM head from pretrained GPT-2.
        
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'gpt2-medium')
        """
        from transformers import GPT2LMHeadModel
        
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Get vocab size and d_model from HF model
        vocab_size = hf_model.config.vocab_size
        d_model = hf_model.config.n_embd
        
        # Create our LM head
        lm_head = cls(d_model=d_model, vocab_size=vocab_size, bias=False)
        
        # Copy weights from HF model
        # In GPT-2, lm_head shares weights with wte (token embedding)
        lm_head.lm_head.weight = nn.Parameter(
            hf_model.transformer.wte.weight.clone()
        )
        
        return lm_head