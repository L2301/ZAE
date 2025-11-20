import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import pickle


def load_gpt_model(checkpoint_path):
    """Load pretrained GPT model components or initialize from scratch."""
    from modelcore import GPTCore
    from embedding import Embedding
    
    checkpoint_path = Path(checkpoint_path)
    
    # Default config
    config = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
        'dropout': 0.0,
        'bias': True,
        'vocab_size': 50257
    }
    
    if checkpoint_path.exists():
        # Load from checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config.update(checkpoint.get('config', {}))
        
        gpt_core = GPTCore(
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            block_size=config['block_size'],
            dropout=config['dropout'],
            bias=config['bias']
        )
        
        embedding = Embedding(
            vocab_size=config['vocab_size'],
            n_embd=config['n_embd'],
            block_size=config['block_size']
        )
        
        gpt_core.load_state_dict(checkpoint['gpt_core'])
        embedding.load_state_dict(checkpoint['embedding'])
    else:
        # Initialize from scratch
        print(f"Checkpoint not found. Initializing new model with GPT-2 Small config")
        
        gpt_core = GPTCore(
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            block_size=config['block_size'],
            dropout=config['dropout'],
            bias=config['bias']
        )
        
        embedding = Embedding(
            vocab_size=config['vocab_size'],
            n_embd=config['n_embd'],
            block_size=config['block_size']
        )
        
        # Try to load pretrained GPT-2 weights from HuggingFace
        try:
            print("Attempting to load pretrained GPT-2 weights from HuggingFace...")
            from transformers import GPT2LMHeadModel
            
            hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
            hf_sd = hf_model.state_dict()
            
            # Map HuggingFace state dict to our format
            # GPTCore mapping
            core_sd = {}
            for i in range(config['n_layer']):
                # Attention
                core_sd[f'h.{i}.ln_1.weight'] = hf_sd[f'transformer.h.{i}.ln_1.weight']
                core_sd[f'h.{i}.ln_1.bias'] = hf_sd[f'transformer.h.{i}.ln_1.bias']
                core_sd[f'h.{i}.attn.c_attn.weight'] = hf_sd[f'transformer.h.{i}.attn.c_attn.weight']
                core_sd[f'h.{i}.attn.c_attn.bias'] = hf_sd[f'transformer.h.{i}.attn.c_attn.bias']
                core_sd[f'h.{i}.attn.c_proj.weight'] = hf_sd[f'transformer.h.{i}.attn.c_proj.weight']
                core_sd[f'h.{i}.attn.c_proj.bias'] = hf_sd[f'transformer.h.{i}.attn.c_proj.bias']
                
                # MLP
                core_sd[f'h.{i}.ln_2.weight'] = hf_sd[f'transformer.h.{i}.ln_2.weight']
                core_sd[f'h.{i}.ln_2.bias'] = hf_sd[f'transformer.h.{i}.ln_2.bias']
                core_sd[f'h.{i}.mlp.c_fc.weight'] = hf_sd[f'transformer.h.{i}.mlp.c_fc.weight']
                core_sd[f'h.{i}.mlp.c_fc.bias'] = hf_sd[f'transformer.h.{i}.mlp.c_fc.bias']
                core_sd[f'h.{i}.mlp.c_proj.weight'] = hf_sd[f'transformer.h.{i}.mlp.c_proj.weight']
                core_sd[f'h.{i}.mlp.c_proj.bias'] = hf_sd[f'transformer.h.{i}.mlp.c_proj.bias']
            
            core_sd['ln_f.weight'] = hf_sd['transformer.ln_f.weight']
            core_sd['ln_f.bias'] = hf_sd['transformer.ln_f.bias']
            
            gpt_core.load_state_dict(core_sd)
            
            # Embedding mapping
            emb_sd = {
                'wte.weight': hf_sd['transformer.wte.weight'],
                'wpe.weight': hf_sd['transformer.wpe.weight']
            }
            embedding.load_state_dict(emb_sd)
            
            print("Successfully loaded GPT-2 weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using random initialization")
    
    gpt_core.eval()
    embedding.eval()
    
    return gpt_core, embedding


def extract_hidden_states(gpt_core, embedding, input_ids):
    """
    Extract final hidden states before the language modeling head.
    
    input_ids: (batch_size, seq_len)
    returns: (hidden_states, token_embeddings)
    """
    with torch.no_grad():
        # Get token-only embeddings (without positions)
        tok_emb = embedding.token_only(input_ids)  # (B, T, n_embd)
        
        # Get full embeddings (with positions) for transformer
        full_emb = embedding(input_ids)  # (B, T, n_embd)
        
        # Forward through transformer
        hidden_states = gpt_core(full_emb)  # (B, T, n_embd)
        
        return hidden_states, tok_emb


def generate_training_data(
    gpt_core,
    embedding,
    dataset_path,
    output_dir,
    seq_length=128,
    batch_size=32,
    max_samples=100000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate training data pairs: (input_embeddings, hidden_states).
    
    Args:
        gpt_core: Pretrained GPTCore model
        embedding: Embedding module
        dataset_path: Path to tokenized dataset (numpy memmap or similar)
        output_dir: Where to save training data
        seq_length: Sequence length for chunks
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gpt_core = gpt_core.to(device)
    embedding = embedding.to(device)
    gpt_core.eval()
    embedding.eval()
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
    
    # Calculate number of chunks
    n_chunks = min(len(data) // seq_length, max_samples)
    print(f"Generating {n_chunks} training samples...")
    
    # Storage
    all_input_embeddings = []
    all_hidden_states = []
    
    # Generate samples in batches
    for i in tqdm(range(0, n_chunks, batch_size)):
        batch_end = min(i + batch_size, n_chunks)
        actual_batch_size = batch_end - i
        
        # Extract batch of sequences
        batch_sequences = []
        for j in range(i, batch_end):
            start_idx = j * seq_length
            end_idx = start_idx + seq_length
            
            if end_idx > len(data):
                break
                
            seq = data[start_idx:end_idx]
            batch_sequences.append(seq)
        
        if not batch_sequences:
            break
            
        # Convert to tensor
        input_ids = torch.tensor(np.array(batch_sequences), dtype=torch.long, device=device)
        
        # Extract hidden states and input embeddings
        hidden_states, input_embeddings = extract_hidden_states(gpt_core, embedding, input_ids)
        
        # Move to CPU and store (exclude last token's hidden state used for prediction)
        all_input_embeddings.append(input_embeddings[:, :-1, :].cpu())
        all_hidden_states.append(hidden_states[:, :-1, :].cpu())
    
    # Concatenate all batches
    print("Concatenating batches...")
    input_embeddings = torch.cat(all_input_embeddings, dim=0)
    hidden_states = torch.cat(all_hidden_states, dim=0)
    
    print(f"Generated {input_embeddings.shape[0]} samples")
    print(f"Input embeddings shape: {input_embeddings.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Save data
    print("Saving training data...")
    torch.save({
        'input_embeddings': input_embeddings,
        'hidden_states': hidden_states,
        'seq_length': seq_length - 1,  # -1 because we exclude last token
        'n_samples': input_embeddings.shape[0]
    }, output_dir / 'training_data.pt')
    
    # Save metadata
    metadata = {
        'seq_length': seq_length - 1,
        'n_samples': int(input_embeddings.shape[0]),
        'd_model': int(input_embeddings.shape[2]),
        'dataset_path': str(dataset_path)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training data saved to {output_dir}")
    
    return input_embeddings, hidden_states


def load_training_data(data_dir):
    """Load pre-generated training data."""
    data_dir = Path(data_dir)
    
    data = torch.load(data_dir / 'training_data.pt')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return data['input_embeddings'], data['hidden_states'], metadata


class SequenceEncoderDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sequence encoder training."""
    
    def __init__(self, input_embeddings, hidden_states):
        self.input_embeddings = input_embeddings
        self.hidden_states = hidden_states
        
    def __len__(self):
        return len(self.input_embeddings)
    
    def __getitem__(self, idx):
        return {
            'input_embeddings': self.input_embeddings[idx],
            'hidden_states': self.hidden_states[idx]
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to GPT checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to tokenized dataset')
    parser.add_argument('--output_dir', type=str, default='data/encoder_training', help='Output directory')
    parser.add_argument('--seq_length', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=100000, help='Max samples to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading GPT model...")
    gpt_core, embedding = load_gpt_model(args.model_path)
    
    # Generate training data
    generate_training_data(
        gpt_core=gpt_core,
        embedding=embedding,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device
    )