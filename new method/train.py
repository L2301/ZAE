import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import uuid
from datetime import datetime

from encoder import SequenceEncoder
from data import (
    SequenceEncoderDataset, 
    load_training_data,
    generate_training_data,
    load_gpt_model
)


def train_encoder(
    gpt_checkpoint_path,
    dataset_path,
    output_dir='train/new',
    n_epochs=10,
    batch_size=64,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=1000,
    # Data generation params
    seq_length=128,
    max_samples=100000,
    data_batch_size=32,
    # Loss weights
    contrastive_weight=1.0,
    variance_weight=0.1,
    repulsion_weight=0.5,
    repulsion_margin=0.1,
    # Optional: use pre-generated data
    pregenerated_data_dir=None
):
    """
    Train sequence encoder. Generates training data if not provided.
    
    Args:
        gpt_checkpoint_path: Path to GPT checkpoint (for data generation and vocab)
        dataset_path: Path to tokenized dataset for data generation
        output_dir: Directory for checkpoints
        pregenerated_data_dir: Optional pre-generated data dir (skips generation)
    """
    # Create run directory
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Load GPT model for vocab embeddings
    print("Loading GPT model...")
    gpt_core, embedding = load_gpt_model(gpt_checkpoint_path)
    vocab_embeddings = embedding.wte.weight.data.to(device)
    
    # Generate or load training data
    if pregenerated_data_dir:
        print(f"Loading pre-generated data from {pregenerated_data_dir}...")
        input_embeddings, hidden_states, metadata = load_training_data(pregenerated_data_dir)
    else:
        print("Generating training data...")
        temp_data_dir = run_dir / 'data' / 'generated'
        temp_data_dir.mkdir(exist_ok=True)
        
        generate_training_data(
            gpt_core=gpt_core,
            embedding=embedding,
            dataset_path=dataset_path,
            output_dir=temp_data_dir,
            seq_length=seq_length,
            batch_size=data_batch_size,
            max_samples=max_samples,
            device=device
        )
        
        input_embeddings, hidden_states, metadata = load_training_data(temp_data_dir)
    
    # Create dataset and dataloader
    dataset = SequenceEncoderDataset(input_embeddings, hidden_states)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    d_model = metadata['d_model']
    encoder = SequenceEncoder(d_model=d_model).to(device)
    vocab_embeddings = vocab_embeddings.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(dataloader) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training state
    global_step = 0
    training_log = []
    
    # Save training config
    config = {
        'run_id': run_id,
        'gpt_checkpoint_path': str(gpt_checkpoint_path),
        'dataset_path': str(dataset_path),
        'd_model': d_model,
        'seq_length': seq_length,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'contrastive_weight': contrastive_weight,
        'variance_weight': variance_weight,
        'repulsion_weight': repulsion_weight,
        'repulsion_margin': repulsion_margin,
        'total_samples': len(dataset),
        'max_samples_generated': max_samples
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training encoder on {len(dataset)} samples for {n_epochs} epochs")
    print(f"Run ID: {run_id}")
    
    # Training loop
    encoder.train()
    for epoch in range(n_epochs):
        epoch_losses = {'total': 0, 'contrastive': 0, 'variance': 0, 'repulsion': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_emb = batch['input_embeddings'].to(device)  # (B, seq_len, d_model)
            hidden = batch['hidden_states'].to(device)  # (B, seq_len, d_model)
            
            # Encode both input and hidden sequences
            encoded_input = encoder(input_emb)  # (B, d_model)
            encoded_hidden = encoder(hidden)  # (B, d_model)
            
            # Compute losses
            contrastive_loss = encoder.compute_contrastive_loss(encoded_input, encoded_hidden)
            variance_loss = encoder.compute_variance_loss(encoded_input)
            repulsion_loss = encoder.compute_vocab_repulsion_loss(
                encoded_input, vocab_embeddings, margin=repulsion_margin
            )
            
            # Combined loss
            total_loss = (
                contrastive_weight * contrastive_loss +
                variance_weight * variance_loss +
                repulsion_weight * repulsion_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['contrastive'] += contrastive_loss.item()
            epoch_losses['variance'] += variance_loss.item()
            epoch_losses['repulsion'] += repulsion_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'contr': f"{contrastive_loss.item():.4f}",
                'var': f"{variance_loss.item():.4f}",
                'rep': f"{repulsion_loss.item():.4f}"
            })
            
            # Logging
            if global_step % log_interval == 0:
                log_entry = {
                    'step': global_step,
                    'epoch': epoch,
                    'total_loss': total_loss.item(),
                    'contrastive_loss': contrastive_loss.item(),
                    'variance_loss': variance_loss.item(),
                    'repulsion_loss': repulsion_loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                }
                training_log.append(log_entry)
            
            # Save checkpoint
            if global_step % save_interval == 0 and global_step > 0:
                checkpoint_path = run_dir / 'model' / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }, checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
            
            global_step += 1
        
        # Epoch summary
        n_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_losses['total']/n_batches:.4f}")
        print(f"  Contrastive: {epoch_losses['contrastive']/n_batches:.4f}")
        print(f"  Variance: {epoch_losses['variance']/n_batches:.4f}")
        print(f"  Repulsion: {epoch_losses['repulsion']/n_batches:.4f}")
    
    # Save final model
    final_path = run_dir / 'model' / 'final_model.pt'
    torch.save({
        'step': global_step,
        'epoch': n_epochs,
        'model_state_dict': encoder.state_dict(),
        'config': config
    }, final_path)
    
    # Save training log
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete. Saved to {run_dir}")
    return encoder, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt_checkpoint', type=str, required=True, help='Path to GPT checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to tokenized dataset')
    parser.add_argument('--output_dir', type=str, default='train/new')
    parser.add_argument('--pregenerated_data', type=str, default=None, help='Optional pre-generated data dir')
    
    # Training params
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data generation params
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=100000)
    parser.add_argument('--data_batch_size', type=int, default=32)
    
    # Loss weights
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    parser.add_argument('--variance_weight', type=float, default=0.1)
    parser.add_argument('--repulsion_weight', type=float, default=0.5)
    parser.add_argument('--repulsion_margin', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Train
    train_encoder(
        gpt_checkpoint_path=args.gpt_checkpoint,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
        contrastive_weight=args.contrastive_weight,
        variance_weight=args.variance_weight,
        repulsion_weight=args.repulsion_weight,
        repulsion_margin=args.repulsion_margin,
        pregenerated_data_dir=args.pregenerated_data
    )