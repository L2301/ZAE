import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from encoder import SequenceEncoder
from decoder import SequenceDecoder
from model.modelcore import GPTCore
from model.tokenembedandun import Embedding
from model.lmhead import LMHead


def load_full_model(checkpoint_path='github', device='cuda'):
    """Load encoder, GPT core, decoder, embedding, and LM head."""
    
    if checkpoint_path == 'github':
        # Download from GitHub release
        import urllib.request
        url = "https://github.com/L2301/ZAE/releases/download/final_modelAdamW1e55epochs/final_model.pt"
        checkpoint_path = 'checkpoints/final_model_adamw.pt'
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not Path(checkpoint_path).exists():
            print(f"Downloading model from {url}...")
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Saved model to {checkpoint_path}")
        else:
            print(f"Using cached model at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load encoder
    encoder = SequenceEncoder(d_model=768).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    
    # Load GPT core
    gpt_core = GPTCore(n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.0, bias=True).to(device)
    gpt_core.load_state_dict(checkpoint['gpt_core_state_dict'])
    gpt_core.eval()
    
    # Load decoder
    decoder = SequenceDecoder(d_model=768).to(device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    # Load embedding from GPT-2
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
    
    # Load LM head
    lm_head = LMHead.from_pretrained_gpt2('gpt2').to(device)
    
    return encoder, gpt_core, decoder, embedding, lm_head


def evaluate_on_fineweb(
    checkpoint_path,
    device='cuda',
    seq_length=4,
    num_samples=10000,
    batch_size=32,
    split='train',
    streaming=True
):
    """Evaluate model on FineWeb dataset."""
    
    print(f"Loading model from {checkpoint_path}...")
    encoder, gpt_core, decoder, embedding, lm_head = load_full_model(checkpoint_path, device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Load FineWeb dataset
    print(f"Loading FineWeb dataset (streaming={streaming})...")
    dataset = load_dataset(
        'HuggingFaceFW/fineweb',
        name='sample-10BT',  # Use the 10B token sample
        split=split,
        streaming=streaming
    )
    
    # Evaluation loop
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    current_batch_embs = []
    current_batch_ids = []
    
    print(f"Evaluating on {num_samples} samples...")
    pbar = tqdm(total=num_samples)
    
    samples_processed = 0
    for example in dataset:
        if samples_processed >= num_samples:
            break
        
        # Tokenize text without special tokens (matching training)
        text = example['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Skip if empty
        if len(tokens) == 0:
            continue
        
        # Take or pad to exact seq_length
        if len(tokens) >= seq_length:
            tokens = tokens[:seq_length]
        else:
            # Pad with a padding strategy: repeat last token to fill
            # (better than EOS since model wasn't trained with EOS)
            tokens = tokens + [tokens[-1]] * (seq_length - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        
        # Get embeddings
        with torch.no_grad():
            tok_emb = embedding.token_only(input_ids.unsqueeze(0))
        
        current_batch_embs.append(tok_emb.squeeze(0))
        current_batch_ids.append(input_ids)
        
        # Process batch when full
        if len(current_batch_embs) == batch_size:
            with torch.no_grad():
                # Stack batch
                emb_batch = torch.stack(current_batch_embs)
                ids_batch = torch.stack(current_batch_ids)
                
                # Encode
                compressed = encoder(emb_batch)  # (B, 768)
                
                # Through GPT
                compressed_unsqueezed = compressed.unsqueeze(1)
                gpt_hidden = gpt_core(compressed_unsqueezed).squeeze(1)
                
                # Decode
                reconstructed = decoder(gpt_hidden, target_seq_len=seq_length)
                
                # LM loss
                pred_logits = lm_head(reconstructed)
                lm_loss = F.cross_entropy(
                    pred_logits.reshape(-1, pred_logits.size(-1)),
                    ids_batch.reshape(-1)
                )
                
                # Accuracy
                preds = pred_logits.argmax(dim=-1)
                acc = (preds == ids_batch).float().mean()
                
                total_loss += lm_loss.item()
                total_acc += acc.item()
                num_batches += 1
                
                samples_processed += batch_size
                pbar.update(batch_size)
                pbar.set_postfix({
                    'loss': f"{total_loss/num_batches:.4f}",
                    'acc': f"{total_acc/num_batches:.3f}"
                })
            
            # Reset batch
            current_batch_embs = []
            current_batch_ids = []
    
    # Process remaining samples in incomplete batch
    if current_batch_embs:
        with torch.no_grad():
            emb_batch = torch.stack(current_batch_embs)
            ids_batch = torch.stack(current_batch_ids)
            
            compressed = encoder(emb_batch)
            compressed_unsqueezed = compressed.unsqueeze(1)
            gpt_hidden = gpt_core(compressed_unsqueezed).squeeze(1)
            reconstructed = decoder(gpt_hidden, target_seq_len=seq_length)
            
            pred_logits = lm_head(reconstructed)
            lm_loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                ids_batch.reshape(-1)
            )
            
            preds = pred_logits.argmax(dim=-1)
            acc = (preds == ids_batch).float().mean()
            
            total_loss += lm_loss.item()
            total_acc += acc.item()
            num_batches += 1
            
            samples_processed += len(current_batch_embs)
            pbar.update(len(current_batch_embs))
    
    pbar.close()
    
    # Print results
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    print("\n" + "="*50)
    print("FineWeb Evaluation Results")
    print("="*50)
    print(f"Samples evaluated: {samples_processed}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Accuracy: {avg_acc:.3f}")
    print("="*50)
    
    return avg_loss, avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='github',
                        help='Path to model checkpoint or "github" to download')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_length', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--no_streaming', action='store_true',
                        help='Disable streaming mode')
    
    args = parser.parse_args()
    
    evaluate_on_fineweb(
        checkpoint_path=args.checkpoint,
        device=args.device,
        seq_length=args.seq_length,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        split=args.split,
        streaming=not args.no_streaming
    )