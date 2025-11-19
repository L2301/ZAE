"""
Mixed training: Alternates between vector reconstruction and language modeling
Configurable ratio (e.g., 5:1 means 5 reconstruction tasks per 1 LM task)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
from datetime import datetime
import uuid
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import ModelHarness


def load_wikitext(tokenizer, seq_len, batch_size):
    """Load wikitext dataset and create dataloader"""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    def tokenize_batch(batch_size, seq_len):
        """Generator that yields tokenized batches"""
        buffer = []
        for example in dataset:
            tokens = tokenizer.encode(example['text'], max_length=seq_len, truncation=True)
            if len(tokens) == seq_len:
                buffer.append(tokens)
                if len(buffer) == batch_size:
                    yield torch.tensor(buffer)
                    buffer = []
    
    return tokenize_batch(batch_size, seq_len)


def train_reconstruction_task(harness, optimizer, device, vocab_size, n_embd, batch_size, seq_len, steps):
    """Single reconstruction training task"""
    harness.train()
    
    losses = []
    pbar = tqdm(range(steps), desc="Reconstruction", leave=False)
    
    for step in pbar:
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            embedding_vectors = harness.embedding.wte(token_ids)
        
        reconstructed = harness.vector_encoder.forward(embedding_vectors.detach())
        x = harness.embedding.add_positions(reconstructed)
        hidden = harness.gpt_core(x)
        final_reconstructed = harness.vector_decoder.forward(hidden)
        
        sse_loss = ((final_reconstructed - embedding_vectors) ** 2).sum()
        cos_sim = nn.functional.cosine_similarity(
            final_reconstructed.view(-1, n_embd),
            embedding_vectors.view(-1, n_embd),
            dim=-1
        ).mean()
        cos_loss = 1 - cos_sim
        loss = (0.015 * sse_loss) + cos_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return sum(losses) / len(losses)


def train_lm_task(harness, optimizer, device, vocab_size, data_iter, seq_len, steps):
    """Single language modeling training task"""
    harness.train()
    
    losses = []
    pbar = tqdm(range(steps), desc="LM", leave=False)
    
    for step in pbar:
        try:
            token_ids = next(data_iter).to(device)
        except StopIteration:
            break
        
        x = harness.embedding(token_ids)
        hidden = harness.gpt_core(x)
        logits = harness.unembedding(hidden)
        
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            token_ids[:, 1:].reshape(-1),
            ignore_index=-1
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return sum(losses) / len(losses)


def train_mixed(
    # Training config
    reconstruction_tasks=5,
    lm_tasks=1,
    total_task_batches=100,
    steps_per_task=100,
    # Model params
    vocab_size=50257,
    n_embd=768,
    vector_bottleneck=128,
    # Task-specific params
    reconstruction_batch_size=32,
    reconstruction_seq_len=1,
    reconstruction_lr=1e-4,
    lm_batch_size=32,
    lm_seq_len=64,
    lm_lr=3e-4,
    # Checkpointing
    checkpoint_dir='checkpoints/train',
    device=None,
):
    """
    Train with mixed tasks
    
    Args:
        reconstruction_tasks: Number of reconstruction tasks per cycle
        lm_tasks: Number of LM tasks per cycle
        total_task_batches: How many full cycles to run
        steps_per_task: Steps per individual task
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    
    run_id = f"mixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(checkpoint_dir, run_id)
    data_dir = os.path.join(run_dir, 'data')
    model_dir = os.path.join(run_dir, 'model')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Task ratio: {reconstruction_tasks} reconstruction : {lm_tasks} LM")
    print(f"Total cycles: {total_task_batches}")
    
    # Initialize
    harness = ModelHarness(
        vocab_size=vocab_size,
        n_embd=n_embd,
        vector_bottleneck=vector_bottleneck,
        use_hard_routing=True,
        hard_route_class=0,
    ).to(device)
    
    harness.load_gpt2_weights('gpt2')
    harness.unfreeze_all()  # Unfreeze once at start
    optimizer = optim.AdamW(harness.parameters(), lr=3e-4)
    
    # Load dataset for LM
    print("Loading Wikitext dataset...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    data_iter = load_wikitext(tokenizer, lm_seq_len, lm_batch_size)
    
    # Training loop
    task_history = []
    total_tasks = (reconstruction_tasks + lm_tasks) * total_task_batches
    
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for cycle in range(total_task_batches):
            # Reconstruction tasks
            for param_group in optimizer.param_groups:
                param_group['lr'] = reconstruction_lr
            
            for i in range(reconstruction_tasks):
                avg_loss = train_reconstruction_task(
                    harness, optimizer, device, vocab_size, n_embd,
                    reconstruction_batch_size, reconstruction_seq_len, steps_per_task
                )
                task_history.append({
                    'cycle': cycle,
                    'type': 'reconstruction',
                    'avg_loss': avg_loss
                })
                pbar.update(1)
                pbar.set_postfix({'cycle': cycle, 'last_loss': f'{avg_loss:.4f}'})
            
            # LM tasks
            for param_group in optimizer.param_groups:
                param_group['lr'] = lm_lr
            
            for i in range(lm_tasks):
                avg_loss = train_lm_task(
                    harness, optimizer, device, vocab_size, data_iter,
                    lm_seq_len, steps_per_task
                )
                task_history.append({
                    'cycle': cycle,
                    'type': 'lm',
                    'avg_loss': avg_loss
                })
                pbar.update(1)
                pbar.set_postfix({'cycle': cycle, 'last_loss': f'{avg_loss:.4f}'})
    
    # Save
    training_data = {
        'run_id': run_id,
        'config': {
            'reconstruction_tasks': reconstruction_tasks,
            'lm_tasks': lm_tasks,
            'total_task_batches': total_task_batches,
            'steps_per_task': steps_per_task,
            'reconstruction_lr': reconstruction_lr,
            'lm_lr': lm_lr,
        },
        'task_history': task_history,
    }
    
    with open(os.path.join(data_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f, indent=2)
    
    torch.save({
        'model_state_dict': harness.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, 'checkpoint.pt'))
    
    print(f"\nSaved to {run_dir}")
    return harness, training_data


if __name__ == '__main__':
    train_mixed(
        reconstruction_tasks=5,
        lm_tasks=1,
        total_task_batches=100,
        steps_per_task=100,
    )