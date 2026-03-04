"""
Layer-wise probing of trained language models.

A linear probe is a small classifier trained on frozen hidden states to test
whether a specific layer encodes a target property (e.g. "how many a's have
we seen so far?"). If a probe trained on layer k achieves high accuracy,
layer k encodes that property.

Barry's notes:
  - Do probing for each layer separately
  - Lower layers = local patterns, higher layers = abstract/counting structure
  - Ignore PAD tokens; keep CLS + SEP
  - Don't split in-dist/out-dist for probing — probe diagnostic, not evaluation

Usage:
    python probe.py --checkpoint checkpoints/anbn_lstm.pt \\
                    --data data/anbn_n1-20.txt \\
                    --grammar grammars/anbn.txt

Output: per-layer probe accuracy + heatmap saved to figures/
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from grammar_loader import load_grammar, build_vocab, tokenize
from train import GrammarDataset, collate_fn, LSTMLanguageModel, TransformerLanguageModel


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_lstm(model: LSTMLanguageModel, batch: torch.Tensor):
    """
    Run a forward pass and collect hidden states at each LSTM layer.

    Returns:
        List of (B, T, H) tensors, one per layer.
    """
    model.eval()
    with torch.no_grad():
        _, layer_hiddens = model(batch, return_hidden=True)
    return layer_hiddens


def extract_hidden_states_transformer(model: TransformerLanguageModel, batch: torch.Tensor):
    """
    Run a forward pass and collect hidden states after each Transformer layer.

    Returns:
        List of (B, T, H) tensors, one per layer.

    Uses forward hooks to capture intermediate representations.
    """
    model.eval()
    layer_hiddens = []
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # TransformerEncoderLayer output is (B, T, H)
            layer_hiddens.append(output.detach())
        return hook

    for i, layer in enumerate(model.transformer.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        B, T = batch.shape
        positions = torch.arange(T, device=batch.device).unsqueeze(0)
        emb = model.embedding(batch) + model.pos_embedding(positions)
        pad_mask = (batch == 0)
        causal = None
        if model.causal_mask:
            causal = nn.Transformer.generate_square_subsequent_mask(T, device=batch.device)
        model.transformer(emb, mask=causal, src_key_padding_mask=pad_mask)

    for h in hooks:
        h.remove()

    return layer_hiddens


# ---------------------------------------------------------------------------
# Probe target: "current symbol count" (simplified example)
# ---------------------------------------------------------------------------

def make_probe_labels(batch: torch.Tensor, vocab: dict) -> torch.Tensor:
    """
    Example probe target: at each position, how many 'a' tokens have been seen so far?

    This directly tests whether the model tracks the counting property of a^n b^n.
    Returns (B, T) integer labels.

    TODO: extend to other structural properties (e.g. "which phase: a-run or b-run?")
    """
    a_id = vocab.get('a', -1)
    labels = torch.zeros_like(batch)
    for b_idx in range(batch.size(0)):
        count = 0
        for t_idx in range(batch.size(1)):
            if batch[b_idx, t_idx].item() == a_id:
                count += 1
            labels[b_idx, t_idx] = count
    return labels


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """A single linear layer trained on frozen hidden states."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_probe(hidden_states: torch.Tensor, labels: torch.Tensor,
                pad_mask: torch.Tensor, num_epochs: int = 20) -> float:
    """
    Train a linear probe on hidden_states and return final accuracy.

    hidden_states : (B, T, H)
    labels        : (B, T) integer class labels
    pad_mask      : (B, T) bool, True where token is PAD — excluded from loss
    """
    H = hidden_states.size(-1)
    num_classes = int(labels.max().item()) + 1

    probe = LinearProbe(H, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Flatten and mask out PAD positions
    mask = ~pad_mask                                    # (B, T) valid positions
    h_flat = hidden_states[mask]                        # (N, H)
    l_flat = labels[mask]                               # (N,)

    for _ in range(num_epochs):
        logits = probe(h_flat)
        loss = criterion(logits, l_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = probe(h_flat).argmax(dim=-1)
        accuracy = (preds == l_flat).float().mean().item()

    return accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Layer-wise probing of a trained model.')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint from train.py')
    parser.add_argument('--data',       required=True, help='Grammar strings (one per line)')
    parser.add_argument('--grammar',    required=True, help='Grammar definition file')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--probe-epochs', type=int, default=20)
    parser.add_argument('--figures-dir', default='figures')
    args = parser.parse_args()

    device = torch.device('cpu')    # probing runs on CPU for simplicity

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab = ckpt['vocab']
    model_type = ckpt['model_type']
    model_args = ckpt['args']
    pad_id = vocab['[PAD]']

    # Rebuild model
    vocab_size = len(vocab)
    if model_type == 'lstm':
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=model_args['embed_dim'],
            hidden_dim=model_args['hidden_dim'],
            num_layers=model_args['num_layers'],
        )
    else:
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            embed_dim=model_args['embed_dim'],
            num_heads=4,
            num_layers=model_args['num_layers'],
            ff_dim=model_args['hidden_dim'],
            causal_mask=ckpt['causal_mask'],
        )

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Data
    grammar = load_grammar(args.grammar)
    dataset = GrammarDataset(args.data, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # Probe each layer
    # Accumulate hidden states + labels across all batches
    all_layer_hiddens = None
    all_labels = []
    all_pad_masks = []

    for batch in dataloader:
        batch = batch.to(device)
        pad_mask = (batch == pad_id)

        if model_type == 'lstm':
            layer_hiddens = extract_hidden_states_lstm(model, batch)
        else:
            layer_hiddens = extract_hidden_states_transformer(model, batch)

        labels = make_probe_labels(batch, vocab)

        if all_layer_hiddens is None:
            all_layer_hiddens = [[] for _ in layer_hiddens]

        for i, h in enumerate(layer_hiddens):
            all_layer_hiddens[i].append(h)
        all_labels.append(labels)
        all_pad_masks.append(pad_mask)

    # Concatenate across batches
    all_layer_hiddens = [torch.cat(hs, dim=0) for hs in all_layer_hiddens]
    all_labels = torch.cat(all_labels, dim=0)
    all_pad_masks = torch.cat(all_pad_masks, dim=0)

    # Train a probe per layer and report accuracy
    print(f"\nLayer-wise probe accuracy ({ckpt['grammar_name']} / {model_type})")
    print("-" * 40)
    accuracies = []
    for layer_idx, hidden in enumerate(all_layer_hiddens):
        acc = train_probe(hidden, all_labels, all_pad_masks, num_epochs=args.probe_epochs)
        print(f"  Layer {layer_idx + 1}: {acc:.4f}")
        accuracies.append(acc)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(accuracies) + 1), accuracies)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Probe accuracy')
    ax.set_title(f'{ckpt["grammar_name"]} — {model_type}')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, len(accuracies) + 1))
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = f'{args.figures_dir}/{ckpt["grammar_name"]}_{model_type}_probe.png'
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"\nFigure saved to {out_path}")


if __name__ == '__main__':
    main()
