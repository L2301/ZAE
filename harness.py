"""
Model Harness - Orchestrates all components with flexible configuration
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class ModelHarness(nn.Module):
    """
    Orchestrates GPT-2 core, embeddings, encoders/decoders, and router
    """
    
    def __init__(
        self,
        # Core model params
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50257,
        dropout=0.0,
        bias=True,
        # Vector encoder/decoder params
        vector_bottleneck=128,
        # Sequence encoder/decoder params
        seq_length=4,
        seq_bottleneck=256,
        use_attention_seq=True,  # True for attention, False for flat
        # Router params
        router_hidden=256,
        # Hard-coded routing (for phases 1.0 and 1.25)
        use_hard_routing=False,
        hard_route_class=None,  # 0, 1, or 2 when use_hard_routing=True
        # Component imports
        modelcore_module=None,
        embeddings_module=None,
        router_module=None,
        vector_encoder_module=None,
        vector_decoder_module=None,
        seq_encoder_module=None,
        seq_decoder_module=None,
    ):
        super().__init__()
        
        # Import modules
        from importlib import import_module
        if modelcore_module is None:
            modelcore = import_module('model.modelcore')
        if embeddings_module is None:
            embeddings = import_module('model.tokenembedandun')
        if router_module is None:
            router = import_module('ZAE.router')
        if vector_encoder_module is None:
            if use_attention_seq:
                vec_enc = import_module('ZAE.vector.encoder')
                vec_dec = import_module('ZAE.vector.decoder')
                seq_enc = import_module('ZAE.sequence.encoderatten')
                seq_dec = import_module('ZAE.sequence.decoderatten')
            else:
                vec_enc = import_module('ZAE.vector.encoder')
                vec_dec = import_module('ZAE.vector.decoder')
                seq_enc = import_module('ZAE.sequence.encoder')
                seq_dec = import_module('ZAE.sequence.decoder')
        
        # Store config
        self.n_embd = n_embd
        self.use_hard_routing = use_hard_routing
        self.hard_route_class = hard_route_class
        
        # Initialize components
        self.embedding = embeddings.Embedding(vocab_size, n_embd, block_size)
        self.gpt_core = modelcore.GPTCore(n_layer, n_head, n_embd, block_size, dropout, bias)
        self.unembedding = embeddings.Unembedding(vocab_size, n_embd)
        
        # Router
        self.router = router.GumbelClassifier(n_embd, router_hidden, num_classes=3)
        
        # Vector encoder/decoder
        self.vector_encoder = vec_enc.VectorEncoder(n_embd, vector_bottleneck)
        self.vector_decoder = vec_dec.VectorDecoder(n_embd, vector_bottleneck)
        
        # Sequence encoder/decoder
        if use_attention_seq:
            self.seq_encoder = seq_enc.SequenceEncoder(seq_length, n_embd, seq_bottleneck)
            self.seq_decoder = seq_dec.SequenceDecoder(seq_length, n_embd, seq_bottleneck)
        else:
            self.seq_encoder = seq_enc.FlatSequenceEncoder(seq_length, n_embd, seq_bottleneck)
            self.seq_decoder = seq_dec.FlatSequenceDecoder(seq_length, n_embd, seq_bottleneck)
        
        self.seq_length = seq_length
    
    def load_gpt2_weights(self, model_type='gpt2'):
        """Load pretrained GPT-2 weights into embedding, core, and unembedding"""
        print(f"Loading {model_type} weights...")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # Load embeddings
        self.embedding.wte.weight.data.copy_(sd_hf['transformer.wte.weight'])
        self.embedding.wpe.weight.data.copy_(sd_hf['transformer.wpe.weight'])
        
        # Load unembedding
        self.unembedding.lm_head.weight.data.copy_(sd_hf['lm_head.weight'])
        
        # Load core transformer blocks
        sd = self.gpt_core.state_dict()
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for key in sd.keys():
            if key.endswith('.attn.bias'):
                continue
            hf_key = f'transformer.{key}'
            if hf_key in sd_hf:
                if any(key.endswith(w) for w in transposed):
                    sd[key].copy_(sd_hf[hf_key].t())
                else:
                    sd[key].copy_(sd_hf[hf_key])
        
        print("Loaded GPT-2 weights successfully")
    
    def freeze_component(self, component_name):
        """Freeze a component (no gradient updates)"""
        component = getattr(self, component_name)
        for param in component.parameters():
            param.requires_grad = False
        print(f"Froze {component_name}")
    
    def unfreeze_component(self, component_name):
        """Unfreeze a component (enable gradient updates)"""
        component = getattr(self, component_name)
        for param in component.parameters():
            param.requires_grad = True
        print(f"Unfroze {component_name}")
    
    def freeze_all(self):
        """Freeze all components"""
        for param in self.parameters():
            param.requires_grad = False
        print("Froze all components")
    
    def unfreeze_all(self):
        """Unfreeze all components"""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfroze all components")
    
    def token_to_vector_embedding(self, idx):
        """
        Convert token to vector embedding (compressed representation)
        
        Args:
            idx: (batch, seq_len) token indices
        Returns:
            (batch, seq_len, bottleneck_dim) compressed vectors
        """
        tok_emb = self.embedding.token_only(idx)  # (batch, seq_len, 768)
        return self.vector_encoder.encode(tok_emb)
    
    def vector_embedding_to_model_input(self, z_vec):
        """
        Decode vector embeddings and add positions
        
        Args:
            z_vec: (batch, seq_len, bottleneck_dim) compressed vectors
        Returns:
            (batch, seq_len, 768) ready for GPT core
        """
        tok_emb = self.vector_decoder.decode(z_vec)  # (batch, seq_len, 768)
        return self.embedding.add_positions(tok_emb)
    
    def sequence_to_vector_embedding(self, x):
        """
        Compress a sequence of vectors into single vector
        
        Args:
            x: (batch, seq_len, 768) sequence of vectors
        Returns:
            (batch, bottleneck_dim) compressed sequence representation
        """
        return self.seq_encoder.encode(x)
    
    def vector_embedding_to_sequence(self, z_seq):
        """
        Expand single vector into sequence
        
        Args:
            z_seq: (batch, bottleneck_dim) compressed sequence
        Returns:
            (batch, seq_len, 768) reconstructed sequence
        """
        return self.seq_decoder.decode(z_seq)
    
    def route_vector(self, x, tau=1.0, hard=False):
        """
        Route a vector to class {0, 1, 2}
        
        Args:
            x: (batch, 768) vector to route
            tau: temperature for Gumbel-Softmax
            hard: if True, returns one-hot; if False, returns soft
        Returns:
            (batch, 3) routing distribution or one-hot
        """
        if self.use_hard_routing:
            # Hard-coded routing
            batch_size = x.size(0)
            route = torch.zeros(batch_size, 3, device=x.device)
            route[:, self.hard_route_class] = 1.0
            return route
        else:
            return self.router(x, tau, hard)
    
    def classify_vector(self, x):
        """
        Deterministic classification
        
        Args:
            x: (batch, 768) vector
        Returns:
            (batch,) class indices {0, 1, 2}
        """
        if self.use_hard_routing:
            batch_size = x.size(0)
            return torch.full((batch_size,), self.hard_route_class, device=x.device)
        else:
            return self.router.classify(x)
    
    def forward_standard(self, idx, targets=None):
        """
        Standard GPT-2 forward pass (no routing)
        
        Args:
            idx: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens (optional)
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar (if targets provided)
        """
        x = self.embedding(idx)
        x = self.gpt_core(x)
        logits = self.unembedding(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    def forward_with_routing(self, idx, tau=1.0, hard=False, return_routes=False):
        """
        Forward pass with vector/sequence routing
        
        Args:
            idx: (batch, seq_len) token indices
            tau: temperature for routing
            hard: hard routing decision
            return_routes: if True, return routing decisions
        Returns:
            logits: (batch, seq_len, vocab_size)
            routes: (batch, seq_len, 3) routing distributions (if return_routes=True)
        """
        # Get embeddings
        x = self.embedding(idx)  # (batch, seq_len, 768)
        
        # Route each vector
        batch_size, seq_len, _ = x.size()
        x_flat = x.view(-1, self.n_embd)  # (batch*seq_len, 768)
        routes = self.route_vector(x_flat, tau, hard)  # (batch*seq_len, 3)
        routes = routes.view(batch_size, seq_len, 3)  # (batch, seq_len, 3)
        
        # Process based on routing (simplified - routes to be implemented per training phase)
        x = self.gpt_core(x)
        logits = self.unembedding(x)
        
        if return_routes:
            return logits, routes
        return logits
    
    def get_config(self):
        """Return current configuration"""
        return {
            'n_embd': self.n_embd,
            'seq_length': self.seq_length,
            'use_hard_routing': self.use_hard_routing,
            'hard_route_class': self.hard_route_class,
        }