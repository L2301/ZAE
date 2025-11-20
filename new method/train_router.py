import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from tqdm import tqdm
import uuid
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from router import GumbelClassifier
from encoder import SequenceEncoder
from data import SequenceEncoderDataset
import urllib.request


import urllib.request


def download_encoder_from_github(output_path='checkpoints/encoder_from_github.pt'):
    """Download pretrained encoder from GitHub releases."""
    url = "https://github.com/L2301/ZAE/releases/download/v0.1/final_model.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"Using cached encoder at {output_path}")
        return str(output_path)
    
    print(f"Downloading encoder from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved encoder to {output_path}")
    return str(output_path)


class RouterDataset(Dataset):
    """Dataset that generates pairs of (vector, label) for router training.
    
    Label mapping:
        0: normal token (raw embedding)
        1: sequence-compressed token
    """
    
    def __init__(self, gpt_core, embedding, encoder, dataset_path, 
                 seq_length=4, max_samples=100000, device='cpu'):
        self.gpt_core = gpt_core.to(device)
        self.embedding = embedding.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.seq_length = seq_length
        
        self.gpt_core.eval()
        self.embedding.eval()
        self.encoder.eval()
        
        # Load tokenized dataset
        if dataset_path is None or not Path(dataset_path).exists():
            from data import download_and_tokenize_wikitext
            dataset_path = Path(download_and_tokenize_wikitext())
        
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        self.n_samples = min(len(self.data) // seq_length, max_samples)
    
    def __len__(self):
        return self.n_samples * 2  # Each sample generates 2 examples (normal + compressed)
    
    def __getitem__(self, idx):
        """Generate training example on-the-fly."""
        # Map idx to actual sequence index and label
        seq_idx = idx // 2
        label = idx % 2  # 0 = normal, 1 = compressed
        
        # Extract sequence
        start_idx = seq_idx * self.seq_length
        end_idx = start_idx + self.seq_length
        seq = self.data[start_idx:end_idx]
        input_ids = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get embeddings
            tok_emb = self.embedding.token_only(input_ids)  # (1, seq_len, 768)
            
            if label == 0:
                # Normal token: pick random position
                pos = torch.randint(0, tok_emb.shape[1], (1,)).item()
                vector = tok_emb[0, pos, :].cpu()
            else:
                # Compressed token: encode the sequence
                vector = self.encoder(tok_emb).cpu().squeeze(0)
        
        return {
            'vector': vector,
            'label': label
        }


def train_router(
    encoder_checkpoint_path,
    gpt_checkpoint_path,
    dataset_path=None,
    output_dir='train/router',
    n_epochs=5,
    batch_size=128,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    seq_length=4,
    max_samples=500000,
    tau_start=1.0,
    tau_end=0.1
):
    """Train router to classify normal vs sequence-compressed tokens."""
    
    run_id = f"router_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Load GPT model
    print("Loading GPT model...")
    from train import load_gpt_model
    gpt_core, embedding = load_gpt_model(gpt_checkpoint_path)
    gpt_core.to(device)
    embedding.to(device)
    
    # Load trained encoder
    if encoder_checkpoint_path is None or encoder_checkpoint_path == 'github':
        print("No encoder checkpoint provided, downloading from GitHub...")
        encoder_checkpoint_path = download_encoder_from_github()
    
    print(f"Loading encoder from {encoder_checkpoint_path}...")
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
    encoder = SequenceEncoder(d_model=768).to(device)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder.eval()
    
    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Create dataset
    print("Creating dataset...")
    import numpy as np
    dataset = RouterDataset(
        gpt_core=gpt_core,
        embedding=embedding,
        encoder=encoder,
        dataset_path=dataset_path,
        seq_length=seq_length,
        max_samples=max_samples,
        device=device
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize router
    router = GumbelClassifier(input_dim=768, hidden_dim=256, num_classes=2).to(device)
    
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * n_epochs
    )
    
    # Save config
    config = {
        'run_id': run_id,
        'encoder_checkpoint_path': str(encoder_checkpoint_path),
        'gpt_checkpoint_path': str(gpt_checkpoint_path),
        'dataset_path': str(dataset_path) if dataset_path else 'auto-downloaded',
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seq_length': seq_length,
        'max_samples': max_samples,
        'tau_start': tau_start,
        'tau_end': tau_end,
        'total_samples': len(dataset)
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training router on {len(dataset)} samples for {n_epochs} epochs")
    print(f"Run ID: {run_id}")
    
    # Training loop
    router.train()
    global_step = 0
    training_log = []
    
    # Early stopping
    low_loss_counter = 0
    low_loss_threshold = 0.1
    low_loss_steps = 200
    
    # Temperature annealing schedule
    total_steps = len(dataloader) * n_epochs
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            vectors = batch['vector'].to(device)
            labels = batch['label'].to(device)
            
            # Anneal temperature
            progress = global_step / total_steps
            tau = tau_start * (tau_end / tau_start) ** progress
            
            # Forward pass with Gumbel-Softmax
            logits_input = router.fc2(torch.relu(router.fc1(vectors)))
            
            # Cross-entropy loss (uses raw logits)
            loss = nn.functional.cross_entropy(logits_input, labels)
            
            # Accuracy
            preds = torch.argmax(logits_input, dim=-1)
            acc = (preds == labels).float().mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            # Early stopping check
            if loss.item() < low_loss_threshold:
                low_loss_counter += 1
                if low_loss_counter >= low_loss_steps:
                    print(f"\n Early stopping triggered! Loss below {low_loss_threshold} for {low_loss_steps} steps.")
                    break
            else:
                low_loss_counter = 0
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}",
                'tau': f"{tau:.3f}"
            })
            
            if global_step % log_interval == 0:
                training_log.append({
                    'step': global_step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'accuracy': acc.item(),
                    'tau': tau,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            global_step += 1
        
        # Check if early stopping was triggered
        if low_loss_counter >= low_loss_steps:
            break
        
        n_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {epoch_loss/n_batches:.4f}")
        print(f"  Accuracy: {epoch_acc/n_batches:.4f}")
    
    # Save final model
    final_path = run_dir / 'model' / 'router_final.pt'
    torch.save({
        'step': global_step,
        'epoch': n_epochs,
        'model_state_dict': router.state_dict(),
        'config': config
    }, final_path)
    
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete. Saved to {run_dir}")
    return router, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_checkpoint', type=str, default='github',
                        help='Path to trained encoder checkpoint (default: download from GitHub)')
    parser.add_argument('--gpt_checkpoint', type=str, default='checkpoints/gpt_model.pt')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='train/router')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=500000)
    
    args = parser.parse_args()
    
    train_router(
        encoder_checkpoint_path=args.encoder_checkpoint,
        gpt_checkpoint_path=args.gpt_checkpoint,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seq_length=args.seq_length,
        max_samples=args.max_samples
    )