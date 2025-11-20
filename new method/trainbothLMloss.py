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

sys.path.append(str(Path(__file__).parent.parent))

from encoder import SequenceEncoder
from decoder import SequenceDecoder
from ZAE.router import GumbelClassifier
from model.lmhead import LMHead
from data import download_and_tokenize_wikitext
import urllib.request

def download_encoder_from_github(output_path='checkpoints/encoder_from_github.pt'):
    """Download pretrained encoder from GitHub releases."""
    url = "https://github.com/L2301/ZAE/releases/download/chora-snapshot-2025-11-20/final_model.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"Using cached encoder at {output_path}")
        return str(output_path)
    
    print(f"Downloading encoder from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved encoder to {output_path}")
    return str(output_path)


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
    
    return gpt_core, embedding


class JointTrainingDataset(Dataset):
    """Dataset for joint decoder + GPT training."""
    
    def __init__(self, encoder, embedding, dataset_path, seq_length=4, max_samples=1000000, device='cpu'):
        self.encoder = encoder.to(device)
        self.embedding = embedding.to(device)
        self.encoder.eval()
        self.embedding.eval()
        self.device = device
        self.seq_length = seq_length
        
        # Freeze encoder and embedding
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Load dataset
        if dataset_path is None or not Path(dataset_path).exists():
            dataset_path = Path(download_and_tokenize_wikitext())
        
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        self.n_samples = min(len(self.data) // seq_length, max_samples)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Generate training example on-the-fly."""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        seq = self.data[start_idx:end_idx]
        input_ids = torch.tensor(seq, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Get embeddings and encode
            tok_emb = self.embedding.token_only(input_ids.unsqueeze(0))  # (1, seq_len, 768)
            compressed = self.encoder(tok_emb)  # (1, 768)
        
        return {
            'compressed': compressed.cpu().squeeze(0),
            'input_ids': input_ids.cpu()
        }


def train_joint_ce(
    encoder_checkpoint_path='github',
    router_checkpoint_path=None,
    gpt_checkpoint_path='checkpoints/gpt_model.pt',
    dataset_path=None,
    output_dir='train/joint_ce',
    n_epochs=10,
    batch_size=32,
    gpt_lr=1e-5,
    decoder_lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=1000,
    seq_length=4,
    max_samples=500000,
    router_weight=1.0,
    lm_weight=1.0
):
    """Train decoder and GPT jointly with cross-entropy loss."""
    
    run_id = f"joint_ce_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Load encoder
    if encoder_checkpoint_path is None or encoder_checkpoint_path == 'github':
        print("Downloading encoder from GitHub...")
        encoder_checkpoint_path = download_encoder_from_github()
    
    print(f"Loading encoder from {encoder_checkpoint_path}...")
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
    encoder = SequenceEncoder(d_model=768).to(device)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Load router
    if router_checkpoint_path is None:
        raise ValueError("Must provide router checkpoint path")
    
    print(f"Loading router from {router_checkpoint_path}...")
    router_checkpoint = torch.load(router_checkpoint_path, map_location=device)
    router = GumbelClassifier(input_dim=768, hidden_dim=256, num_classes=2).to(device)
    router.load_state_dict(router_checkpoint['model_state_dict'])
    router.eval()
    for param in router.parameters():
        param.requires_grad = False
    
    # Load GPT models
    print("Loading GPT models...")
    gpt_core_trainable, embedding = load_gpt_model(gpt_checkpoint_path)
    gpt_core_trainable.to(device)
    embedding.to(device)
    
    # Freeze embedding
    embedding.eval()
    for param in embedding.parameters():
        param.requires_grad = False
    
    # Initialize decoder and LM head
    decoder = SequenceDecoder(d_model=768).to(device)
    lm_head = LMHead.from_pretrained_gpt2('gpt2').to(device)
    
    # Create dataset
    print("Creating dataset...")
    dataset = JointTrainingDataset(
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
    
    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': gpt_core_trainable.parameters(), 'lr': gpt_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr}
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * n_epochs)
    
    # Save config
    config = {
        'run_id': run_id,
        'encoder_checkpoint_path': str(encoder_checkpoint_path),
        'router_checkpoint_path': str(router_checkpoint_path),
        'gpt_checkpoint_path': str(gpt_checkpoint_path),
        'dataset_path': str(dataset_path) if dataset_path else 'auto-downloaded',
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'gpt_lr': gpt_lr,
        'decoder_lr': decoder_lr,
        'seq_length': seq_length,
        'max_samples': max_samples,
        'router_weight': router_weight,
        'lm_weight': lm_weight,
        'total_samples': len(dataset)
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training on {len(dataset)} samples for {n_epochs} epochs")
    print(f"Run ID: {run_id}")
    print(f"GPT LR: {gpt_lr}, Decoder LR: {decoder_lr}")
    
    # Training loop
    gpt_core_trainable.train()
    decoder.train()
    global_step = 0
    training_log = []
    
    for epoch in range(n_epochs):
        epoch_losses = {'total': 0, 'router': 0, 'lm': 0}
        epoch_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            compressed = batch['compressed'].to(device)  # (B, 768)
            input_ids = batch['input_ids'].to(device)  # (B, seq_len)
            
            # Forward: compressed -> GPT -> decoder -> reconstructed
            compressed_unsqueezed = compressed.unsqueeze(1)  # (B, 1, 768)
            gpt_hidden = gpt_core_trainable(compressed_unsqueezed)  # (B, 1, 768)
            
            # Router consistency loss: GPT output should be classified as "sequence"
            router_logits = router.fc2(F.relu(router.fc1(gpt_hidden.squeeze(1))))
            target_labels = torch.ones(router_logits.shape[0], dtype=torch.long, device=device)
            router_loss = F.cross_entropy(router_logits, target_labels)
            
            # Decode
            reconstructed = decoder(gpt_hidden.squeeze(1), target_seq_len=seq_length)  # (B, seq_len, 768)
            
            # Get logits and compute cross-entropy on tokens
            pred_logits = lm_head(reconstructed)  # (B, seq_len, vocab_size)
            lm_loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                input_ids.reshape(-1)
            )
            
            # Token accuracy
            preds = pred_logits.argmax(dim=-1)
            acc = (preds == input_ids).float().mean()
            
            # Total loss
            total_loss = router_weight * router_loss + lm_weight * lm_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt_core_trainable.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['router'] += router_loss.item()
            epoch_losses['lm'] += lm_loss.item()
            epoch_losses['repr'] += representation_loss.item()
            epoch_acc += acc.item()
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'router': f"{router_loss.item():.4f}",
                'lm': f"{lm_loss.item():.4f}",
                'repr': f"{representation_loss.item():.4f}",
                'acc': f"{acc.item():.3f}"
            })
            
            if global_step % log_interval == 0:
                training_log.append({
                    'step': global_step,
                    'epoch': epoch,
                    'total_loss': total_loss.item(),
                    'router_loss': router_loss.item(),
                    'lm_loss': lm_loss.item(),
                    'representation_loss': representation_loss.item(),
                    'accuracy': acc.item(),
                    'gpt_lr': optimizer.param_groups[0]['lr'],
                    'decoder_lr': optimizer.param_groups[1]['lr']
                })
            
            if global_step % save_interval == 0 and global_step > 0:
                checkpoint_path = run_dir / 'model' / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'gpt_core_state_dict': gpt_core_trainable.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }, checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
            
            global_step += 1
        
        n_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_losses['total']/n_batches:.4f}")
        print(f"  Router Loss: {epoch_losses['router']/n_batches:.4f}")
        print(f"  LM Loss: {epoch_losses['lm']/n_batches:.4f}")
        print(f"  Repr Loss: {epoch_losses['repr']/n_batches:.4f}")
        print(f"  Accuracy: {epoch_acc/n_batches:.3f}")
    
    # Save final
    final_path = run_dir / 'model' / 'final_model.pt'
    torch.save({
        'step': global_step,
        'epoch': n_epochs,
        'gpt_core_state_dict': gpt_core_trainable.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'config': config
    }, final_path)
    
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete. Saved to {run_dir}")
    return gpt_core_trainable, decoder, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_checkpoint', type=str, default='github')
    parser.add_argument('--router_checkpoint', type=str, required=True)
    parser.add_argument('--gpt_checkpoint', type=str, default='checkpoints/gpt_model.pt')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='train/joint_ce')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpt_lr', type=float, default=1e-5)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=500000)
    parser.add_argument('--router_weight', type=float, default=1.0)
    parser.add_argument('--lm_weight', type=float, default=1.0)
    
    args = parser.parse_args()
    
    train_joint_ce(
        encoder_checkpoint_path=args.encoder_checkpoint,
        router_checkpoint_path=args.router_checkpoint,
        gpt_checkpoint_path=args.gpt_checkpoint,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gpt_lr=args.gpt_lr,
        decoder_lr=args.decoder_lr,
        device=args.device,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        router_weight=args.router_weight,
        lm_weight=args.lm_weight
    )