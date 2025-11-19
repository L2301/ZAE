"""
Phase 1 Training: Vector Encoder/Decoder Reconstruction
- No router (hard-coded type 0)
- Only vector encoder/decoder trainable
- Random token sampling → encode → core (frozen) → decode → reconstruction loss
- Stop when loss < 0.1 for 200 consecutive steps
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import json
import os
import sys
from datetime import datetime
import uuid
from tqdm import tqdm

# Add parent directory to path to import harness
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import ModelHarness


def get_grad_flow_info(harness):
    """Get string showing which components have gradients flowing"""
    components = {
        'embedding': harness.embedding,
        'gpt_core': harness.gpt_core,
        'unembedding': harness.unembedding,
        'router': harness.router,
        'vector_encoder': harness.vector_encoder,
        'vector_decoder': harness.vector_decoder,
        'seq_encoder': harness.seq_encoder,
        'seq_decoder': harness.seq_decoder,
    }
    
    grad_status = []
    for name, component in components.items():
        has_grad = any(p.requires_grad for p in component.parameters())
        if has_grad:
            grad_status.append(f"✓{name}")
    
    return " | ".join(grad_status) if grad_status else "None"


def train_phase1(
    # Model params
    vocab_size=50257,
    n_embd=768,
    vector_bottleneck=128,
    # Training params
    batch_size=32,
    seq_len=1,
    learning_rate=1e-4,
    max_steps=100000,
    eval_interval=100,
    # Early stopping
    target_loss=0.1,
    patience_steps=200,
    # Learning rate scheduler
    warmup_steps=1000,
    min_lr=1e-6,
    # Checkpointing
    checkpoint_dir='checkpoints/train',
    device=None,  # Auto-detect if None
):
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Generate run ID
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(checkpoint_dir, run_id)
    data_dir = os.path.join(run_dir, 'data')
    model_dir = os.path.join(run_dir, 'model')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    
    # Initialize model
    harness = ModelHarness(
        vocab_size=vocab_size,
        n_embd=n_embd,
        vector_bottleneck=vector_bottleneck,
        use_hard_routing=True,
        hard_route_class=0,
    ).to(device)
    
    # Load GPT-2 weights
    harness.load_gpt2_weights('gpt2')
    
    # Freeze everything except vector encoder/decoder
    harness.freeze_all()
    harness.unfreeze_component('vector_encoder')
    harness.unfreeze_component('vector_decoder')
    
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in harness.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Learning rate scheduler: linear warmup + cosine annealing
    # Calculate total steps for cosine annealing (max_steps - warmup_steps)
    cosine_steps = max(1, max_steps - warmup_steps)
    
    # Warmup scheduler: linear increase from 0 to learning_rate
    # At step warmup_steps, we switch to cosine, so warmup should reach 1.0 at step warmup_steps-1
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / max(1, warmup_steps), 1.0) if step < warmup_steps else 1.0
    )
    
    # Cosine annealing scheduler: decay from learning_rate to min_lr
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr
    )
    
    # Sequential scheduler: warmup then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Training state
    step = 0
    total_flops = 0
    training_examples = []
    below_target_count = 0
    losses = []
    
    # Get gradient flow info once
    grad_flow = get_grad_flow_info(harness)
    print(f"Gradient flow: {grad_flow}")
    print("Starting training...")
    
    # Progress bar
    pbar = tqdm(total=max_steps, desc="Phase 1", unit="step")
    
    while step < max_steps:
        harness.train()
        
        # Sample random tokens
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Get embedding vectors (detached - frozen)
        with torch.no_grad():
            embedding_vectors = harness.embedding.wte(token_ids)  # (batch, seq_len, 768)
        
        # Encode-decode BEFORE core (trainable)
        reconstructed = harness.vector_encoder.forward(embedding_vectors.detach())  # 768→bottleneck→768
        
        # Send through frozen core
        x = harness.embedding.add_positions(reconstructed)
        hidden = harness.gpt_core(x)  # (batch, seq_len, 768)
        
        # Encode-decode AFTER core (trainable)
        final_reconstructed = harness.vector_decoder.forward(hidden)  # 768→bottleneck→768
        
        # Loss: reconstruction error + cosine similarity
        sse_loss = ((final_reconstructed - embedding_vectors) ** 2).sum()
        cos_sim = nn.functional.cosine_similarity(
            final_reconstructed.view(-1, n_embd), 
            embedding_vectors.view(-1, n_embd), 
            dim=-1
        ).mean()
        cos_loss = 1 - cos_sim
        
        loss = (0.015 * sse_loss) + cos_loss  # Weight cosine loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        losses.append(loss.item())
        step += 1
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track training examples
        training_examples.append({
            'step': step,
            'tokens': token_ids.cpu().tolist()[:2],  # Save first 2 examples
            'loss': loss.item()
        })
        
        # Estimate FLOPs (rough approximation)
        flops_per_step = batch_size * seq_len * (
            2 * (n_embd * vector_bottleneck * 2 + vector_bottleneck * 2 * vector_bottleneck) +
            2 * (vector_bottleneck * vector_bottleneck * 2 + vector_bottleneck * 2 * n_embd)
        )
        total_flops += flops_per_step
        
        # Check early stopping
        if loss.item() < target_loss:
            below_target_count += 1
            if below_target_count >= patience_steps:
                pbar.close()
                print(f"\nEarly stopping: loss < {target_loss} for {patience_steps} steps")
                break
        else:
            below_target_count = 0
        
        # Update progress bar
        avg_loss = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else sum(losses) / len(losses)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'sse_loss': f'{sse_loss.item():.4f}',
            'cos_loss': f'{cos_loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.2e}',
            'target': f'{below_target_count}/{patience_steps}'
        })
        pbar.update(1)
    
    pbar.close()
    
    print(f"\nTraining complete at step {step}")
    print(f"Final loss: {loss.item():.6f}")
    
    # Save training data
    training_data = {
        'run_id': run_id,
        'config': {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'vector_bottleneck': vector_bottleneck,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'learning_rate': learning_rate,
            'warmup_steps': warmup_steps,
            'min_lr': min_lr,
            'target_loss': target_loss,
            'patience_steps': patience_steps,
        },
        'final_step': step,
        'final_loss': loss.item(),
        'final_lr': optimizer.param_groups[0]['lr'],
        'total_flops': total_flops,
        'training_examples': training_examples,
    }
    
    with open(os.path.join(data_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Saved training data to {data_dir}/training_data.json")
    
    # Save model
    torch.save({
        'step': step,
        'model_state_dict': harness.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
        'config': training_data['config'],
    }, os.path.join(model_dir, 'checkpoint.pt'))
    
    print(f"Saved model to {model_dir}/checkpoint.pt")
    print(f"Total FLOPs: {total_flops:,}")
    
    return harness, training_data


if __name__ == '__main__':
    train_phase1(
        batch_size=32,
        seq_len=64,
        learning_rate=5e-4,
        max_steps=1000000,
        eval_interval=100,
    )