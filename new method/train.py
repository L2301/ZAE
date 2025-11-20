import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import uuid
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ZAE.sequence.encoder import SequenceEncoder
from data import (
    SequenceEncoderDataset, 
    load_training_data,
    generate_training_data,
    download_and_tokenize_wikitext
)


def load_gpt_model(checkpoint_path):
    """Load GPT model or initialize from HuggingFace."""
    from model.modelcore import GPTCore
    from model.tokenembedandun import Embedding
    
    checkpoint_path = Path(checkpoint_path)
    
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
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config.update(checkpoint.get('config', {}))
        
        gpt_core = GPTCore(**{k: config[k] for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'dropout', 'bias']})
        embedding = Embedding(vocab_size=config['vocab_size'], n_embd=config['n_embd'], block_size=config['block_size'])
        
        gpt_core.load_state_dict(checkpoint['gpt_core'])
        embedding.load_state_dict(checkpoint['embedding'])
    else:
        print("Checkpoint not found. Downloading GPT-2 weights from HuggingFace...")
        
        gpt_core = GPTCore(**{k: config[k] for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'dropout', 'bias']})
        embedding = Embedding(vocab_size=config['vocab_size'], n_embd=config['n_embd'], block_size=config['block_size'])
        
        try:
            from transformers import GPT2LMHeadModel
            
            hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
            hf_sd = hf_model.state_dict()
            
            # Map to GPTCore
            core_sd = {}
            for i in range(config['n_layer']):
                core_sd[f'h.{i}.ln_1.weight'] = hf_sd[f'transformer.h.{i}.ln_1.weight']
                core_sd[f'h.{i}.ln_1.bias'] = hf_sd[f'transformer.h.{i}.ln_1.bias']
                core_sd[f'h.{i}.attn.c_attn.weight'] = hf_sd[f'transformer.h.{i}.attn.c_attn.weight'].T
                core_sd[f'h.{i}.attn.c_attn.bias'] = hf_sd[f'transformer.h.{i}.attn.c_attn.bias']
                core_sd[f'h.{i}.attn.c_proj.weight'] = hf_sd[f'transformer.h.{i}.attn.c_proj.weight'].T
                core_sd[f'h.{i}.attn.c_proj.bias'] = hf_sd[f'transformer.h.{i}.attn.c_proj.bias']
                core_sd[f'h.{i}.ln_2.weight'] = hf_sd[f'transformer.h.{i}.ln_2.weight']
                core_sd[f'h.{i}.ln_2.bias'] = hf_sd[f'transformer.h.{i}.ln_2.bias']
                core_sd[f'h.{i}.mlp.c_fc.weight'] = hf_sd[f'transformer.h.{i}.mlp.c_fc.weight'].T
                core_sd[f'h.{i}.mlp.c_fc.bias'] = hf_sd[f'transformer.h.{i}.mlp.c_fc.bias']
                core_sd[f'h.{i}.mlp.c_proj.weight'] = hf_sd[f'transformer.h.{i}.mlp.c_proj.weight'].T
                core_sd[f'h.{i}.mlp.c_proj.bias'] = hf_sd[f'transformer.h.{i}.mlp.c_proj.bias']
            
            core_sd['ln_f.weight'] = hf_sd['transformer.ln_f.weight']
            core_sd['ln_f.bias'] = hf_sd['transformer.ln_f.bias']
            gpt_core.load_state_dict(core_sd)
            
            # Map to Embedding
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
    """Extract hidden states from GPT."""
    with torch.no_grad():
        tok_emb = embedding.token_only(input_ids)
        full_emb = embedding(input_ids)
        hidden_states = gpt_core(full_emb)
        return hidden_states, tok_emb


def train_encoder(
    gpt_checkpoint_path,
    dataset_path=None,
    output_dir='train/new',
    n_epochs=10,
    batch_size=64,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=1000,
    seq_length=128,
    max_samples=100000,
    data_batch_size=32,
    contrastive_weight=1.0,
    variance_weight=0.1,
    repulsion_weight=0.5,
    repulsion_margin=0.1,
    pregenerated_data_dir=None
):
    """Train sequence encoder."""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Load GPT model
    print("Loading GPT model...")
    gpt_core, embedding = load_gpt_model(gpt_checkpoint_path)
    vocab_embeddings = embedding.wte.weight.data.to(device)
    
    # Generate or load data
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
    
    # Create dataset
    dataset = SequenceEncoderDataset(input_embeddings, hidden_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize encoder
    d_model = metadata['d_model']
    encoder = SequenceEncoder(d_model=d_model).to(device)
    
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * n_epochs)
    
    # Save config
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
    global_step = 0
    training_log = []
    
    for epoch in range(n_epochs):
        epoch_losses = {'total': 0, 'contrastive': 0, 'variance': 0, 'repulsion': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_emb = batch['input_embeddings'].to(device)
            hidden = batch['hidden_states'].to(device)
            
            # Encode
            encoded_input = encoder(input_emb)
            encoded_hidden = encoder(hidden)
            
            # Losses
            contrastive_loss = encoder.compute_contrastive_loss(encoded_input, encoded_hidden)
            variance_loss = encoder.compute_variance_loss(encoded_input)
            repulsion_loss = encoder.compute_vocab_repulsion_loss(encoded_input, vocab_embeddings, margin=repulsion_margin)
            
            total_loss = (
                contrastive_weight * contrastive_loss +
                variance_weight * variance_loss +
                repulsion_weight * repulsion_loss
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['contrastive'] += contrastive_loss.item()
            epoch_losses['variance'] += variance_loss.item()
            epoch_losses['repulsion'] += repulsion_loss.item()
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'contr': f"{contrastive_loss.item():.4f}",
                'var': f"{variance_loss.item():.4f}",
                'rep': f"{repulsion_loss.item():.4f}"
            })
            
            if global_step % log_interval == 0:
                training_log.append({
                    'step': global_step,
                    'epoch': epoch,
                    'total_loss': total_loss.item(),
                    'contrastive_loss': contrastive_loss.item(),
                    'variance_loss': variance_loss.item(),
                    'repulsion_loss': repulsion_loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                })
            
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
        
        n_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_losses['total']/n_batches:.4f}")
        print(f"  Contrastive: {epoch_losses['contrastive']/n_batches:.4f}")
        print(f"  Variance: {epoch_losses['variance']/n_batches:.4f}")
        print(f"  Repulsion: {epoch_losses['repulsion']/n_batches:.4f}")
    
    # Save final
    final_path = run_dir / 'model' / 'final_model.pt'
    torch.save({
        'step': global_step,
        'epoch': n_epochs,
        'model_state_dict': encoder.state_dict(),
        'config': config
    }, final_path)
    
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete. Saved to {run_dir}")
    return encoder, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt_checkpoint', type=str, default='checkpoints/gpt_model.pt')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='train/new')
    parser.add_argument('--pregenerated_data', type=str, default=None)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=100000)
    parser.add_argument('--data_batch_size', type=int, default=32)
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    parser.add_argument('--variance_weight', type=float, default=0.1)
    parser.add_argument('--repulsion_weight', type=float, default=0.5)
    parser.add_argument('--repulsion_margin', type=float, default=0.1)
    
    args = parser.parse_args()
    
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