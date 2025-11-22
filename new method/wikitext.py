import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from encoder import SequenceEncoder
from decoder import SequenceDecoder
from model.modelcore import GPTCore
from model.tokenembedandun import Embedding
from model.lmhead import LMHead
from data import download_and_tokenize_wikitext


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
    
    print(f"Loading checkpoint from {checkpoint_path}...")
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
    
    print("Model loaded successfully!")
    return encoder, gpt_core, decoder, embedding, lm_head


def evaluate_on_wikitext(
    checkpoint_path='github',
    device='cuda',
    seq_length=4,
    num_samples=10000,
    batch_size=32
):
    """Evaluate model on WikiText dataset."""
    
    print(f"Loading model...")
    encoder, gpt_core, decoder, embedding, lm_head = load_full_model(checkpoint_path, device)
    
    # Load WikiText data
    print("Loading WikiText dataset...")
    dataset_path = Path(download_and_tokenize_wikitext())
    data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
    
    # Calculate number of batches
    n_samples = min(num_samples, len(data) // seq_length)
    n_batches = n_samples // batch_size
    
    print(f"Evaluating on {n_samples} samples ({n_batches} batches)...")
    
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(range(n_batches), desc="Evaluating")
    
    for batch_idx in pbar:
        batch_embs = []
        batch_ids = []
        
        for i in range(batch_size):
            sample_idx = batch_idx * batch_size + i
            start_idx = sample_idx * seq_length
            end_idx = start_idx + seq_length
            
            seq = data[start_idx:end_idx]
            input_ids = torch.tensor(seq, dtype=torch.long, device=device)
            
            # Get embeddings
            with torch.no_grad():
                tok_emb = embedding.token_only(input_ids.unsqueeze(0))
            
            batch_embs.append(tok_emb.squeeze(0))
            batch_ids.append(input_ids)
        
        # Process batch
        with torch.no_grad():
            emb_batch = torch.stack(batch_embs)
            ids_batch = torch.stack(batch_ids)
            
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
            
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{total_acc/(batch_idx+1):.3f}"
            })
    
    # Print results
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    
    print("\n" + "="*50)
    print("WikiText Evaluation Results")
    print("="*50)
    print(f"Samples evaluated: {n_samples}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Accuracy: {avg_acc:.3f}")
    print("="*50)
    
    # Also print some example predictions
    print("\nExample predictions (last batch):")
    for i in range(min(3, batch_size)):
        tokens = ids_batch[i].cpu().numpy()
        pred_tokens = preds[i].cpu().numpy()
        
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
        true_text = tokenizer.decode(tokens)
        pred_text = tokenizer.decode(pred_tokens)
        
        print(f"\n  Sample {i+1}:")
        print(f"    True: {true_text}")
        print(f"    Pred: {pred_text}")
    
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
    
    args = parser.parse_args()
    
    evaluate_on_wikitext(
        checkpoint_path=args.checkpoint,
        device=args.device,
        seq_length=args.seq_length,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )