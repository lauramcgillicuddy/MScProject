"""
Train an LSTM or Transformer on grammar strings.

The model is trained as a language model: given a prefix, predict the next token.
Trained representations are later analysed in probe.py.

Usage:
    python train.py --data data/anbn_n1-20.txt --grammar grammars/anbn.txt --model lstm
    python train.py --data data/anbn_n1-20.txt --grammar grammars/anbn.txt --model transformer
    python train.py --data data/anbn_n1-20.txt --grammar grammars/anbn.txt --model transformer --no-causal-mask

Checkpoints are saved to: checkpoints/<grammar_name>_<model_type>.pt
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from grammar_loader import load_grammar, build_vocab, tokenize


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GrammarDataset(Dataset):
    """Loads grammar strings and returns token-ID sequences."""

    def __init__(self, filepath: str, vocab: dict):
        self.vocab = vocab
        self.sequences = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids = tokenize(line, vocab, add_special=True)
                    self.sequences.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch, pad_id: int):
    """Pad sequences in a batch to the same length."""
    max_len = max(seq.size(0) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :seq.size(0)] = seq
    return padded


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LSTMLanguageModel(nn.Module):
    """Vanilla LSTM language model. Hidden states used for probing."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, return_hidden=False):
        """
        x: (batch, seq_len) token IDs
        Returns logits (batch, seq_len, vocab_size).
        If return_hidden=True, also returns all hidden states per layer.
        """
        emb = self.embedding(x)                         # (B, T, E)
        out, _ = self.lstm(emb)                         # (B, T, H)
        logits = self.output_proj(out)                  # (B, T, V)
        if return_hidden:
            return logits, out                          # out = final layer activations
        return logits


class TransformerLanguageModel(nn.Module):
    """
    Transformer language model.

    Use causal_mask=True  for autoregressive (GPT-style) training.
    Use causal_mask=False for bidirectional (BERT-style) training.
    Barry's note: test both — bidirectional should handle closing brackets better.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, ff_dim: int = 256, dropout: float = 0.1,
                 max_seq_len: int = 512, causal_mask: bool = True):
        super().__init__()
        self.causal_mask = causal_mask
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.num_layers = num_layers

    def forward(self, x, return_hidden=False):
        """
        x: (batch, seq_len) token IDs
        Returns logits (batch, seq_len, vocab_size).
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        emb = self.embedding(x) + self.pos_embedding(positions)

        # Padding mask: True where token is PAD (id=0)
        pad_mask = (x == 0)

        # Causal mask: upper triangular, prevents attending to future tokens
        causal = None
        if self.causal_mask:
            causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # TODO: extract per-layer hidden states for probing
        # nn.TransformerEncoder doesn't expose intermediate layers natively.
        # Options: (1) iterate self.transformer.layers manually,
        #          (2) register forward hooks in probe.py.
        out = self.transformer(emb, mask=causal, src_key_padding_mask=pad_mask)
        logits = self.output_proj(out)

        if return_hidden:
            return logits, out
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, dataloader, optimizer, device, pad_id: int):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        # Next-token prediction: input = batch[:, :-1], target = batch[:, 1:]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)                          # (B, T-1, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train a language model on grammar strings.')
    parser.add_argument('--data',    required=True, help='Training data file (one string per line)')
    parser.add_argument('--grammar', required=True, help='Grammar definition file (for vocabulary)')
    parser.add_argument('--model',   choices=['lstm', 'transformer'], default='lstm')
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--embed-dim',  type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--no-causal-mask', action='store_true',
                        help='Disable causal masking (bidirectional Transformer)')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Vocabulary from grammar file
    grammar = load_grammar(args.grammar)
    vocab = build_vocab(grammar)
    vocab_size = len(vocab)
    pad_id = vocab['[PAD]']
    print(f"Vocabulary ({vocab_size} tokens): {vocab}")

    # Dataset
    dataset = GrammarDataset(args.data, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    print(f"Training on {len(dataset)} sequences")

    # Model
    if args.model == 'lstm':
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_heads=4,
            num_layers=args.num_layers,
            ff_dim=args.hidden_dim,
            causal_mask=not args.no_causal_mask,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model: {args.model}  params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    for epoch in range(1, args.epochs + 1):
        loss = train(model, dataloader, optimizer, device, pad_id)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{args.epochs}  loss: {loss:.4f}")

    # Save checkpoint
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    suffix = 'no_causal' if args.no_causal_mask else 'causal'
    tag = args.model if args.model == 'lstm' else f'transformer_{suffix}'
    checkpoint_path = f"{args.checkpoint_dir}/{grammar.name}_{tag}.pt"

    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'grammar_name': grammar.name,
        'model_type': args.model,
        'causal_mask': not args.no_causal_mask,
        'args': vars(args),
    }, checkpoint_path)
    print(f"Saved: {checkpoint_path}")


if __name__ == '__main__':
    main()
