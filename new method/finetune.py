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
    """Download encoder from chora-snapshot-2025-11-20 release."""
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


def download_joint_checkpoint_from_github(output_path='checkpoints/joint_from_github.pt'):
    """Download GPT+decoder from jointdon3.7ready4finetune release."""
    url = "https://github.com/L2301/ZAE/releases/download/jointdon3.7ready4finetune/final_model.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"Using cached joint checkpoint at {output_path}")
        return str(output_path)
    
    print(f"Downloading joint checkpoint from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved joint checkpoint to {output_path}")
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config.update(checkpoint.get('config', {}))
        
        gpt_core = GPTCore(**{k: config[k] for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'dropout', 'bias']})
        embedding = Embedding(vocab_size=config['vocab_size'], n_embd=config['n_embd'], block_size=config['block_size'])
        
        gpt_core.load_state_dict(checkpoint['gpt_core'])
        embedding.load_state_dict(checkpoint['embedding'])
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    return gpt_core, embedding


class EncoderFinetuneDataset(Dataset):
    """Dataset for encoder finetuning."""
    
    def __init__(self, embedding, dataset_path, seq_length=4, max_samples=500000, device='cpu'):
        self.embedding = embedding.to(device)
        self.embedding.eval()
        self.device = device
        self.seq_length = seq_length
        
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        if dataset_path is None or not Path(dataset_path).exists():
            dataset_path = Path(download_and_tokenize_wikitext())
        
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        self.n_samples = min(len(self.data) // seq_length, max_samples)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        seq = self.data[start_idx:end_idx]
        input_ids = torch.tensor(seq, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            tok_emb = self.embedding.token_only(input_ids.unsqueeze(0))
        
        return {
            'embeddings': tok_emb.cpu().squeeze(0),
            'input_ids': input_ids.cpu()
        }


def finetune_encoder(
    joint_checkpoint_path='github',
    dataset_path=None,
    output_dir='train/encoder_finetune',
    n_epochs=5,
    batch_size=32,
    learning_rate=1e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=1000,
    seq_length=4,
    max_samples=500000
):
    """Finetune encoder on LM loss."""
    
    run_id = f"enc_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    
    # Load encoder from chora-snapshot-2025-11-20
    encoder = SequenceEncoder(d_model=768).to(device)
    if joint_checkpoint_path == 'github':
        print("Downloading encoder from chora-snapshot-2025-11-20...")
        enc_path = download_encoder_from_github()
        enc_ckpt = torch.load(enc_path, map_location=device)
        encoder.load_state_dict(enc_ckpt['model_state_dict'])
    else:
        print(f"Loading encoder from custom checkpoint {joint_checkpoint_path}...")
        checkpoint = torch.load(joint_checkpoint_path, map_location=device)
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            # Fallback to GitHub
            enc_path = download_encoder_from_github()
            enc_ckpt = torch.load(enc_path, map_location=device)
            encoder.load_state_dict(enc_ckpt['model_state_dict'])
    
    # Load GPT and decoder from jointdon3.7ready4finetune
    if joint_checkpoint_path == 'github':
        print("Downloading GPT+decoder from jointdon3.7ready4finetune...")
        joint_path = download_joint_checkpoint_from_github()
        joint_checkpoint = torch.load(joint_path, map_location=device)
    else:
        joint_checkpoint = torch.load(joint_checkpoint_path, map_location=device)
    
    from model.modelcore import GPTCore
    gpt_core = GPTCore(n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.0, bias=True).to(device)
    gpt_core.load_state_dict(joint_checkpoint['gpt_core_state_dict'])
    
    decoder = SequenceDecoder(d_model=768).to(device)
    decoder.load_state_dict(joint_checkpoint['decoder_state_dict'])
    
    # Load embedding (frozen) - download GPT-2 if needed
    from transformers import GPT2LMHeadModel
    print("Loading embedding from GPT-2...")
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    from model.tokenembedandun import Embedding
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
    
    # Create dataset
    print("Creating dataset...")
    dataset = EncoderFinetuneDataset(
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
    
    # Optimizer for all three components with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': learning_rate * 10},  # Encoder learns fastest
        {'params': gpt_core.parameters(), 'lr': learning_rate},
        {'params': decoder.parameters(), 'lr': learning_rate * 5}
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * n_epochs)
    
    config = {
        'run_id': run_id,
        'joint_checkpoint_path': str(joint_checkpoint_path),
        'dataset_path': str(dataset_path) if dataset_path else 'auto-downloaded',
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seq_length': seq_length,
        'max_samples': max_samples,
        'total_samples': len(dataset)
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Finetuning encoder on {len(dataset)} samples for {n_epochs} epochs")
    print(f"Run ID: {run_id}")
    
    # Training loop
    encoder.train()
    gpt_core.train()
    decoder.train()
    global_step = 0
    training_log = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            emb = batch['embeddings'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            # Encode
            compressed = encoder(emb)  # (B, 768)
            
            # Through GPT
            compressed_unsqueezed = compressed.unsqueeze(1)
            gpt_hidden = gpt_core(compressed_unsqueezed).squeeze(1)
            
            # Decode
            reconstructed = decoder(gpt_hidden, target_seq_len=seq_length)
            
            # LM loss
            pred_logits = lm_head(reconstructed)
            lm_loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                input_ids.reshape(-1)
            )
            
            # Accuracy
            preds = pred_logits.argmax(dim=-1)
            acc = (preds == input_ids).float().mean()
            
            optimizer.zero_grad()
            lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(gpt_core.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += lm_loss.item()
            epoch_acc += acc.item()
            
            pbar.set_postfix({
                'loss': f"{lm_loss.item():.4f}",
                'acc': f"{acc.item():.3f}"
            })
            
            if global_step % log_interval == 0:
                training_log.append({
                    'step': global_step,
                    'epoch': epoch,
                    'lm_loss': lm_loss.item(),
                    'accuracy': acc.item(),
                    'lr': scheduler.get_last_lr()[0]
                })
            
            if global_step % save_interval == 0 and global_step > 0:
                checkpoint_path = run_dir / 'model' / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'gpt_core_state_dict': gpt_core.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }, checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
            
            global_step += 1
        
        n_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  LM Loss: {epoch_loss/n_batches:.4f}")
        print(f"  Accuracy: {epoch_acc/n_batches:.3f}")
    
    # Save final
    final_path = run_dir / 'model' / 'final_model.pt'
    torch.save({
        'step': global_step,
        'epoch': n_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'gpt_core_state_dict': gpt_core.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'config': config
    }, final_path)
    
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nFinetuning complete. Saved to {run_dir}")
    return encoder, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_checkpoint', type=str, default='github',
                        help='Joint checkpoint path or "github" to download from release')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='train/encoder_finetune')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=500000)
    
    args = parser.parse_args()
    
    finetune_encoder(
        joint_checkpoint_path=args.joint_checkpoint,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seq_length=args.seq_length,
        max_samples=args.max_samples
    )