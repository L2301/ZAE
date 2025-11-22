"""
Train full ZAE system from scratch with PROPER autoregressive context
Multiple compressed sequences through GPT to predict next sequence
SUPPORTS VARIABLE CONTEXT LENGTHS for more robust training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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
from coherence import CoherenceAttention


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


class VariableContextDataset(Dataset):
    """Dataset that provides VARIABLE context of N compressed sequences to predict sequence N+1.
    
    Each sample can have a different context length between min_context and max_context.
    This creates more robust training by exposing the model to varying amounts of context.
    
    Flow:
    chunks [0,1,2,...,N-1] → encoder → [<seq0>, <seq1>, ..., <seqN-1>]
    → GPT → [<seq0>, <seq1>, ..., <seqN-1>, <pred_seqN>]
    → decoder(<pred_seqN>) → chunk_N tokens
    
    where N varies per sample from min_context to max_context
    """
    
    def __init__(self, encoder, embedding, dataset_path, 
                 chunk_size=4, min_context=2, max_context=16, 
                 max_samples=1000000, device='cpu'):
        self.encoder = encoder.to(device)
        self.embedding = embedding.to(device)
        self.encoder.eval()
        self.embedding.eval()
        self.device = device
        self.chunk_size = chunk_size
        self.min_context = min_context
        self.max_context = max_context
    
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Load dataset
        if dataset_path is None or not Path(dataset_path).exists():
            from data import download_and_tokenize_c4
            dataset_path = Path(download_and_tokenize_c4())
        
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        # Need max_context + 1 chunks per sample (max context + target)
        total_tokens_needed = (max_context + 1) * chunk_size
        self.n_samples = min(len(self.data) // total_tokens_needed, max_samples)
        
        print(f"Dataset initialized: {self.n_samples} samples")
        print(f"Context range: {min_context}-{max_context} chunks of {chunk_size} tokens")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Generate training example with VARIABLE context length.
        
        Returns:
            compressed_context: (variable_context_length, 768) - encoded context chunks
            all_target_ids: (variable_context_length, chunk_size) - ALL chunks for autoregressive training
            context_length: int - actual context length for this sample
        """
        # Randomly choose context length for this sample
        context_length = torch.randint(self.min_context, self.max_context + 1, (1,)).item()
        
        start_idx = idx * (self.max_context + 1) * self.chunk_size
        
        compressed_chunks = []
        all_target_chunks = []
        
        # Encode ALL chunks (we need them as targets for autoregressive training)
        for i in range(context_length):
            chunk_start = start_idx + i * self.chunk_size
            chunk_end = chunk_start + self.chunk_size
            chunk_tokens = self.data[chunk_start:chunk_end]
            chunk_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=self.device)
            
            # Compress for input
            with torch.no_grad():
                tok_emb = self.embedding.token_only(chunk_ids.unsqueeze(0))
                compressed = self.encoder(tok_emb).squeeze(0)  # (768,)
            
            compressed_chunks.append(compressed)
            all_target_chunks.append(chunk_ids)
        
        # Stack
        compressed_context = torch.stack(compressed_chunks)  # (context_length, 768)
        all_target_ids = torch.stack(all_target_chunks)  # (context_length, chunk_size)
        
        return {
            'compressed_context': compressed_context.cpu(),
            'all_target_ids': all_target_ids.cpu(),
            'context_length': context_length
        }


def variable_context_collate_fn(batch):
    """Custom collate function to handle variable-length contexts.
    
    Pads sequences to the longest in the batch and creates attention masks.
    
    Args:
        batch: List of dicts with 'compressed_context', 'all_target_ids', 'context_length'
    
    Returns:
        dict with padded tensors and attention masks
    """
    # Extract contexts and targets
    contexts = [item['compressed_context'] for item in batch]
    targets = [item['all_target_ids'] for item in batch]  # List of (seq_len, chunk_size)
    context_lengths = torch.tensor([item['context_length'] for item in batch])
    
    # Pad contexts to max length in batch
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0.0)
    # Result is (batch, max_seq_len, 768)
    
    # Pad targets - need to handle 2D tensors (seq_len, chunk_size)
    max_len = max(t.size(0) for t in targets)
    chunk_size = targets[0].size(1)
    
    padded_targets = torch.full((len(batch), max_len, chunk_size), -100, dtype=torch.long)
    for i, target in enumerate(targets):
        seq_len = target.size(0)
        padded_targets[i, :seq_len, :] = target
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(context_lengths):
        attention_mask[i, :length] = 1
    
    return {
        'compressed_context': padded_contexts,
        'target_ids': padded_targets,  # (B, max_seq_len, chunk_size)
        'attention_mask': attention_mask,
        'context_lengths': context_lengths
    }


def train_autoregressive_system(
    output_dir='train/autoregressive_variable',
    dataset_path=None,
    n_epochs=20,
    batch_size=64,
    encoder_lr=1e-4,
    gpt_lr=5e-5,
    decoder_lr=1e-4,
    coherence_lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    save_interval=10000,
    chunk_size=4,
    min_context=2,
    max_context=16,
    max_samples=1000000,
    use_coherence=True
):
    """Train full system with VARIABLE autoregressive context lengths."""
    
    run_id = f"varctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
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
    print("Creating variable context dataset...")
    dataset = VariableContextDataset(
        encoder=encoder,
        embedding=embedding,
        dataset_path=dataset_path,
        chunk_size=chunk_size,
        min_context=min_context,
        max_context=max_context,
        max_samples=max_samples,
        device=device
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=variable_context_collate_fn
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
    scaler = torch.cuda.amp.GradScaler()
    
    config = {
        'run_id': run_id,
        'pretrained': False,
        'variable_context': True,
        'dataset_path': str(dataset_path) if dataset_path else 'auto-downloaded',
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'encoder_lr': encoder_lr,
        'gpt_lr': gpt_lr,
        'decoder_lr': decoder_lr,
        'coherence_lr': coherence_lr if coherence else None,
        'chunk_size': chunk_size,
        'min_context': min_context,
        'max_context': max_context,
        'max_samples': max_samples,
        'use_coherence': use_coherence,
        'total_samples': len(dataset)
    }
    
    with open(run_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining autoregressive system with VARIABLE context")
    print(f"Run ID: {run_id}")
    print(f"Total samples: {len(dataset)}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Context range: {min_context}-{max_context} chunks of {chunk_size} tokens each")
    print(f"Task: Given [<seq0>, <seq1>, ..., <seqN>] → predict all next sequences autoregressively\n")
    
    # Training loop
    encoder.train()
    gpt_core.train()
    decoder.train()
    if coherence:
        coherence.train()
    
    global_step = 0
    training_log = []
    context_length_stats = {i: 0 for i in range(min_context, max_context + 1)}
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            compressed_context = batch['compressed_context'].to(device)  # (B, max_seq_len, 768)
            target_ids = batch['target_ids'].to(device)  # (B, max_seq_len, chunk_size)
            attention_mask = batch['attention_mask'].to(device)  # (B, max_seq_len)
            context_lengths = batch['context_lengths']
            
            batch_size = compressed_context.size(0)
            
            # Track context length distribution
            for length in context_lengths.tolist():
                context_length_stats[length] += 1
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                # Forward: context through GPT with attention mask
                gpt_output = gpt_core(compressed_context)  # (B, max_seq_len, 768)
                
                # Vectorized autoregressive training
                # Collect all valid prediction positions into a batch
                pred_vectors = []
                pred_targets = []
                
                for i in range(batch_size):
                    seq_len = context_lengths[i].item()
                    # Positions 0 to seq_len-2 predict positions 1 to seq_len-1
                    if seq_len > 1:
                        pred_vectors.append(gpt_output[i, :seq_len-1, :])  # (seq_len-1, 768)
                        pred_targets.append(target_ids[i, 1:seq_len, :])   # (seq_len-1, chunk_size)
                
                if len(pred_vectors) == 0:
                    continue
                    
                # Stack all predictions into single batch
                all_pred_vectors = torch.cat(pred_vectors, dim=0)  # (total_preds, 768)
                all_targets = torch.cat(pred_targets, dim=0)        # (total_preds, chunk_size)
                
                # Decode all at once
                reconstructed = decoder(all_pred_vectors, target_seq_len=chunk_size)  # (total_preds, chunk_size, 768)
                
                if coherence:
                    reconstructed = coherence(reconstructed)
                
                pred_logits = lm_head(reconstructed)  # (total_preds, chunk_size, vocab_size)
                
                # Compute loss
                loss = F.cross_entropy(
                    pred_logits.reshape(-1, pred_logits.size(-1)),
                    all_targets.reshape(-1),
                    ignore_index=-100
                )
            
            # Accuracy (outside autocast for stability)
            preds = pred_logits.argmax(dim=-1)
            valid_mask = all_targets != -100
            correct = (preds[valid_mask] == all_targets[valid_mask]).sum().item()
            total = valid_mask.sum().item()
            acc = correct / total if total > 0 else 0.0
            num_predictions = all_pred_vectors.size(0)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(gpt_core.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            if coherence:
                torch.nn.utils.clip_grad_norm_(coherence.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
            # Show stats including context length range in this batch
            min_ctx = context_lengths.min().item()
            max_ctx = context_lengths.max().item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.3f}",
                'ctx': f"{min_ctx}-{max_ctx}",
                'preds': num_predictions
            })
            
            if global_step % log_interval == 0:
                training_log.append({
                    'step': global_step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'accuracy': acc,
                    'lr': scheduler.get_last_lr()[0],
                    'batch_min_context': min_ctx,
                    'batch_max_context': max_ctx,
                    'batch_avg_context': context_lengths.float().mean().item(),
                    'num_predictions': num_predictions
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
        
        # Show context length distribution for this epoch
        if (epoch + 1) % 5 == 0:
            print(f"  Context length distribution:")
            total_samples = sum(context_length_stats.values())
            for ctx_len in range(min_context, max_context + 1):
                count = context_length_stats[ctx_len]
                pct = 100 * count / total_samples if total_samples > 0 else 0
                print(f"    {ctx_len} chunks: {count} samples ({pct:.1f}%)")
    
    # Save final
    final_path = run_dir / 'model' / 'final_model.pt'
    save_dict = {
        'step': global_step,
        'epoch': n_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'gpt_core_state_dict': gpt_core.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'config': config,
        'context_length_stats': context_length_stats
    }
    if coherence:
        save_dict['coherence_state_dict'] = coherence.state_dict()
    torch.save(save_dict, final_path)
    
    with open(run_dir / 'data' / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Save context length statistics
    with open(run_dir / 'data' / 'context_length_stats.json', 'w') as f:
        json.dump(context_length_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Saved to: {run_dir}")
    print(f"\nFinal context length distribution:")
    total_samples = sum(context_length_stats.values())
    for ctx_len in range(min_context, max_context + 1):
        count = context_length_stats[ctx_len]
        pct = 100 * count / total_samples if total_samples > 0 else 0
        print(f"  {ctx_len} chunks: {count} samples ({pct:.1f}%)")
    print(f"{'='*60}\n")
    
    return encoder, gpt_core, decoder, coherence, run_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ZAE with variable context lengths')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to tokenized dataset')
    parser.add_argument('--output_dir', type=str, default='train/autoregressive_variable',
                        help='Output directory for training runs')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                        help='Learning rate for encoder')
    parser.add_argument('--gpt_lr', type=float, default=1e-4,
                        help='Learning rate for GPT core')
    parser.add_argument('--decoder_lr', type=float, default=1e-4,
                        help='Learning rate for decoder')
    parser.add_argument('--coherence_lr', type=float, default=1e-4,
                        help='Learning rate for coherence attention')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--chunk_size', type=int, default=4,
                        help='Number of tokens per chunk')
    parser.add_argument('--min_context', type=int, default=2,
                        help='Minimum number of context chunks')
    parser.add_argument('--max_context', type=int, default=16,
                        help='Maximum number of context chunks')
    parser.add_argument('--max_samples', type=int, default=1000000,
                        help='Maximum number of samples to use')
    parser.add_argument('--no_coherence', action='store_true',
                        help='Disable coherence attention')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Steps between saving checkpoints')
    
    args = parser.parse_args()
    
    train_autoregressive_system(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        gpt_lr=args.gpt_lr,
        decoder_lr=args.decoder_lr,
        coherence_lr=args.coherence_lr,
        device=args.device,
        chunk_size=args.chunk_size,
        min_context=args.min_context,
        max_context=args.max_context,
        max_samples=args.max_samples,
        use_coherence=not args.no_coherence,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )