"""
Train full ZAE system with coherence attention - Using pretrained encoder/decoder/GPT
Trains to predict NEXT 4 tokens from compressed current 4 tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from tqdm import tqdm
import uuid
from datetime import datetime
import sys
import numpy as np
import urllib.request

sys.path.append(str(Path(__file__).parent.parent))

from encoder import SequenceEncoder
from decoder import SequenceDecoder
from model.modelcore import GPTCore
from model.tokenembedandun import Embedding
from model.lmhead import LMHead
from coherence_single_head import CoherenceAttention


def download_from_github(filename, release_tag, output_path):
    """Download model from GitHub release."""
    url = f"https://github.com/L2301/ZAE/releases/download/{release_tag}/{filename}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"Using cached {filename} at {output_path}")
        return str(output_path)
    
    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")
    return str(output_path)


class NextSequencePredictionDataset(Dataset):
    """Dataset for training to predict NEXT sequence from current sequence.
    
    Each sample: [seq1] -> [seq2] where seq2 immediately follows seq1
    """
    
    def __init__(self, encoder, embedding, dataset_path, seq_length=4, max_samples=1000000, device='cpu'):
        self.encoder = encoder.to(device)
        self.embedding = embedding.to(device)
        self.encoder.eval()
        self.embedding.eval()
        self.device = device
        self.seq_length = seq_length
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Load dataset
        if dataset_path is None or not Path(dataset_path).exists():
            from data import download_and_tokenize_openwebtext
            dataset_path = Path(download_and_tokenize_openwebtext())
        
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        # Need 2 sequences per sample
        self.n_samples = min(len(self.data) // (seq_length * 2), max_samples)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Generate training example: current sequence -> next sequence."""
        start_idx = idx * self.seq_length * 2
        
        # Current sequence (input)
        seq1 = self.data[start_idx:start_idx + self.seq_length]
        input_ids = torch.tensor(seq1, dtype=torch.long, device=self.device)
        
        # Next sequence (target)
        seq2 = self.data[start_idx + self.seq_length:start_idx + self.seq_length * 2]
        target_ids = torch.tensor(seq2, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Encode current sequence
            tok_emb = self.embedding.token_only(input_ids.unsqueeze(0))
            compressed = self.encoder(tok_emb).squeeze(0)
        
        return {
            'compressed': compressed.cpu(),
            'target_ids': target_ids.cpu()
        }


def train_full_system_pretrained(
    output_dir='train/full_system_pretrained',
    dataset_path=None,
    n_epochs=10,
    batch_size=32,
    learning_rate=1e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=10000,
    seq_length=4,
    max_samples=1000000,
    use_coherence=True
):
    """Train full system using pretrained encoder/decoder/GPT."""
    
    run_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Download pretrained components
    print("Downloading pretrained components...")
    enc_path = download_from_github('final_model.pt', 'chora-snapshot-2025-11-20', 'checkpoints/encoder_pretrained.pt')
    joint_path = download_from_github('final_model.pt', 'final_modelAdamW1e55epochs', 'checkpoints/joint_pretrained.pt')
    
    # Load encoder (frozen)
    print("Loading encoder...")
    encoder = SequenceEncoder(d_model=768).to(device)
    enc_ckpt = torch.load(enc_path, map_location=device)
    encoder.load_state_dict(enc_ckpt['model_state_dict'])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Load GPT and decoder (trainable)
    print("Loading GPT and decoder...")
    joint_ckpt = torch.load(joint_path, map_location=device)
    
    gpt_core = GPTCore(n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.0, bias=True).to(device)
    gpt_core.load_state_dict(joint_ckpt['gpt_core_state_dict'])
    
    decoder = SequenceDecoder(d_model=768).to(device)
    decoder.load_state_dict(joint_ckpt['decoder_state_dict'])
    
    # Load embedding (frozen)
    from transformers import GPT2LMHeadModel
    print("Loading embedding from GPT-2...")
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    embedding = Embedding(vocab_size=50257, n_embd=768, block_size=1024).to(device)
    emb_sd = {
        'wte.weight': hf_model.transformer.wte.weight,
        'wpe.weight': hf_model.transformer.wpe.weight
    }
    embedding.load_state_dict(emb_sd)
    embedding.eval()
    for param in embedding.parameters():
        param.requires_grad = False
    
    # Load LM head
    lm_head = LMHead.from_pretrained_gpt2('gpt2').to(device)
    
    # Optional coherence attention
    coherence = None
    if use_coherence:
        print("Initializing coherence attention...")
        coherence = CoherenceAttention(d_model=768, dropout=0.1).to(device)
    
    # Create dataset
    print("Creating dataset...")
    dataset = NextSequencePredictionDataset(
        encoder=encoder,
        embedding=embedding,
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
    
    # Optimizer - train GPT, decoder, and optionally coherence
    trainable_params = [
        {'params': gpt_core.parameters(), 'lr': learning_rate},
        {'params': decoder.parameters(), 'lr': learning_rate * 5}
    ]
    if coherence is not None:
        trainable_params.append({'params': coherence.parameters(), 'lr': learning_rate * 10})
    
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * n_epochs)
    
    config = {
        'run_id': run_id,
        'pretrained': True,
        'encoder_path': str(enc_path),
        'joint_path': str(joint_path),
        'dataset_path': str(dataset_path) if dataset_path else 'auto-downloaded',
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seq_length': seq_length,
        'max_samples': max_samples,
        'use_coherence': use_coherence,
        'total_samples': len(dataset)
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training on {len(dataset)} samples for {n_epochs} epochs")
    print(f"Run ID: {run_id}")
    print(f"Task: Predict NEXT {seq_length} tokens from current {seq_length} tokens")
    
    # Training loop
    gpt_core.train()
    decoder.train()
    if coherence:
        coherence.train()
    
    global_step = 0
    training_log = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            compressed = batch['compressed'].to(device)  # (B, 768)
            target_ids = batch['target_ids'].to(device)  # (B, seq_length)
            
            # Forward: compressed -> GPT -> decoder -> next sequence
            compressed_unsqueezed = compressed.unsqueeze(1)  # (B, 1, 768)
            gpt_hidden = gpt_core(compressed_unsqueezed).squeeze(1)  # (B, 768)
            
            # Decode to next sequence
            reconstructed = decoder(gpt_hidden, target_seq_len=seq_length)  # (B, seq_length, 768)
            
            # Optional coherence attention
            if coherence:
                reconstructed = coherence(reconstructed)
            
            # Predict tokens
            pred_logits = lm_head(reconstructed)  # (B, seq_length, vocab_size)
            
            # Loss
            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            # Accuracy
            preds = pred_logits.argmax(dim=-1)
            acc = (preds == target_ids).float().mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt_core.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            if coherence:
                torch.nn.utils.clip_grad_norm_(coherence.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.3f}"
            })
            
            if global_step % log_interval == 0:
                training_log.append({
                    'step': global_step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'accuracy': acc.item(),
                    'lr': scheduler.get_last_lr()[0]
                })
            
            if global_step % save_interval == 0 and global_step > 0:
                checkpoint_path = run_dir / 'model' / f'checkpoint_step_{global_step}.pt'
                save_dict = {
                    'step': global_step,
                    'epoch': epoch,
                    'gpt_core_state_dict': gpt_core.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }
                if coherence:
                    save_dict['coherence_state_dict'] = coherence.state_dict()
                torch.save(save_dict, checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
            
            global_step += 1
        
        n_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {epoch_loss/n_batches:.4f}")
        print(f"  Accuracy: {epoch_acc/n_batches:.3f}")
    
    # Save final
    final_path = run_dir / 'model' / 'final_model.pt'
    save_dict = {
        'step': global_step,
        'epoch': n_epochs,
        'gpt_core_state_dict': gpt_core.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'config': config
    }
    if coherence:
        save_dict['coherence_state_dict'] = coherence.state_dict()
    torch.save(save_dict, final_path)
    
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete. Saved to {run_dir}")
    return gpt_core, decoder, coherence, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='train/full_system_pretrained')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=1000000)
    parser.add_argument('--no_coherence', action='store_true', help='Disable coherence attention')
    
    args = parser.parse_args()
    
    train_full_system_pretrained(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        use_coherence=not args.no_coherence
    )