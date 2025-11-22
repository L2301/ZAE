"""
Train full ZAE system from scratch - Only GPT-2 embedding and modelcore are pretrained
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

sys.path.append(str(Path(__file__).parent.parent))

from encoder import SequenceEncoder
from decoder import SequenceDecoder
from model.modelcore import GPTCore
from model.tokenembedandun import Embedding
from model.lmhead import LMHead
from coherence_single_head import CoherenceAttention


def load_pretrained_gpt2():
    """Load only GPT-2 embedding and modelcore (everything else random init)."""
    from transformers import GPT2LMHeadModel
    
    print("Loading GPT-2 weights from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    hf_sd = hf_model.state_dict()
    
    # Load GPT core
    gpt_core = GPTCore(n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.0, bias=True)
    
    core_sd = {}
    for i in range(12):
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
    
    # Load embedding
    embedding = Embedding(vocab_size=50257, n_embd=768, block_size=1024)
    emb_sd = {
        'wte.weight': hf_sd['transformer.wte.weight'],
        'wpe.weight': hf_sd['transformer.wpe.weight']
    }
    embedding.load_state_dict(emb_sd)
    
    print("Successfully loaded GPT-2 weights")
    return gpt_core, embedding


class NextSequencePredictionDataset(Dataset):
    """Dataset for training to predict NEXT sequence from current sequence."""
    
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


def train_full_system_scratch(
    output_dir='train/full_system_scratch',
    dataset_path=None,
    n_epochs=20,
    batch_size=32,
    encoder_lr=1e-4,
    gpt_lr=5e-5,
    decoder_lr=1e-4,
    coherence_lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=10000,
    seq_length=4,
    max_samples=1000000,
    use_coherence=True
):
    """Train full system from scratch (except GPT-2 embedding/modelcore)."""
    
    run_id = f"scratch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Load GPT-2 embedding and modelcore
    gpt_core, embedding = load_pretrained_gpt2()
    gpt_core.to(device)
    embedding.to(device)
    
    # Freeze embedding
    embedding.eval()
    for param in embedding.parameters():
        param.requires_grad = False
    
    # Initialize encoder (random)
    print("Initializing encoder (random)...")
    encoder = SequenceEncoder(d_model=768).to(device)
    
    # Initialize decoder (random)
    print("Initializing decoder (random)...")
    decoder = SequenceDecoder(d_model=768).to(device)
    
    # Initialize coherence (random)
    coherence = None
    if use_coherence:
        print("Initializing coherence attention (random)...")
        coherence = CoherenceAttention(d_model=768, dropout=0.1).to(device)
    
    # Load LM head
    lm_head = LMHead.from_pretrained_gpt2('gpt2').to(device)
    
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
    
    # Optimizer - train everything except embedding
    trainable_params = [
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': gpt_core.parameters(), 'lr': gpt_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr}
    ]
    if coherence is not None:
        trainable_params.append({'params': coherence.parameters(), 'lr': coherence_lr})
    
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * n_epochs)
    
    config = {
        'run_id': run_id,
        'pretrained': False,
        'dataset_path': str(dataset_path) if dataset_path else 'auto-downloaded',
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'encoder_lr': encoder_lr,
        'gpt_lr': gpt_lr,
        'decoder_lr': decoder_lr,
        'coherence_lr': coherence_lr if coherence else None,
        'seq_length': seq_length,
        'max_samples': max_samples,
        'use_coherence': use_coherence,
        'total_samples': len(dataset)
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training from scratch on {len(dataset)} samples for {n_epochs} epochs")
    print(f"Run ID: {run_id}")
    print(f"Task: Predict NEXT {seq_length} tokens from current {seq_length} tokens")
    
    # Training loop
    encoder.train()
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
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
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
                    'encoder_state_dict': encoder.state_dict(),
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
        'encoder_state_dict': encoder.state_dict(),
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
    return encoder, gpt_core, decoder, coherence, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='train/full_system_scratch')
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--gpt_lr', type=float, default=5e-5)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--coherence_lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=1000000)
    parser.add_argument('--no_coherence', action='store_true', help='Disable coherence attention')
    
    args = parser.parse_args()
    
    train_full_system_scratch(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        gpt_lr=args.gpt_lr,
        decoder_lr=args.decoder_lr,
        coherence_lr=args.coherence_lr,
        device=args.device,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        use_coherence=not args.no_coherence
    )