import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))


def download_and_tokenize_wikitext(output_path='data/wikitext.bin', split='train'):
    """Download and tokenize wikitext-103."""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"Using cached dataset at {output_path}")
        return str(output_path)
    
    print("Downloading wikitext-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    
    print("Tokenizing...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    all_tokens = []
    for item in tqdm(dataset):
        if item['text'].strip():
            tokens = tokenizer.encode(item['text'])
            all_tokens.extend(tokens)
    
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    
    print(f"Saving {len(tokens_array)} tokens to {output_path}")
    tokens_array.tofile(output_path)
    
    return str(output_path)


def extract_hidden_states(gpt_core, embedding, input_ids):
    """Extract final hidden states before the language modeling head."""
    with torch.no_grad():
        tok_emb = embedding.token_only(input_ids)
        full_emb = embedding(input_ids)
        hidden_states = gpt_core(full_emb)
        return hidden_states, tok_emb


def generate_training_data(
    gpt_core,
    embedding,
    dataset_path=None,
    output_dir='data/encoder_training',
    seq_length=128,
    batch_size=32,
    max_samples=100000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Generate training data pairs: (input_embeddings, hidden_states)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_path is None:
        print("No dataset provided, downloading wikitext-103...")
        dataset_path = download_and_tokenize_wikitext()
    
    gpt_core = gpt_core.to(device)
    embedding = embedding.to(device)
    gpt_core.eval()
    embedding.eval()
    
    print(f"Loading dataset from {dataset_path}")
    data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
    
    n_chunks = min(len(data) // seq_length, max_samples)
    print(f"Generating {n_chunks} training samples...")
    
    all_input_embeddings = []
    all_hidden_states = []
    
    for i in tqdm(range(0, n_chunks, batch_size)):
        batch_end = min(i + batch_size, n_chunks)
        
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
            
        input_ids = torch.tensor(np.array(batch_sequences), dtype=torch.long, device=device)
        hidden_states, input_embeddings = extract_hidden_states(gpt_core, embedding, input_ids)
        
        all_input_embeddings.append(input_embeddings[:, :-1, :].cpu())
        all_hidden_states.append(hidden_states[:, :-1, :].cpu())
    
    print("Concatenating batches...")
    input_embeddings = torch.cat(all_input_embeddings, dim=0)
    hidden_states = torch.cat(all_hidden_states, dim=0)
    
    print(f"Generated {input_embeddings.shape[0]} samples")
    print(f"Input embeddings shape: {input_embeddings.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    print("Saving training data...")
    torch.save({
        'input_embeddings': input_embeddings,
        'hidden_states': hidden_states,
        'seq_length': seq_length - 1,
        'n_samples': input_embeddings.shape[0]
    }, output_dir / 'training_data.pt')
    
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