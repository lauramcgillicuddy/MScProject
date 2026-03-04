"""
Grammar loader and string generator for Artificial Grammar Learning experiments.

Supports two grammar types:
  cfg    - Context-free grammar defined by production rules
  direct - Context-sensitive or fixed patterns (e.g. a:n b:n c:n)

Grammar file format:
    NAME         <name>
    TYPE         cfg | direct
    TERMINALS    <space-separated terminal symbols>
    NONTERMINALS <space-separated non-terminal symbols>  (cfg only)
    START        <start symbol>                          (cfg only)
    PATTERN      <pattern string>                        (direct only)

    RULES
    LHS -> sym1 sym2 ...
    LHS -> sym3 | sym4 sym5

Example (cfg):
    NAME anbn
    TYPE cfg
    NONTERMINALS S
    TERMINALS a b
    START S

    RULES
    S -> a S b
    S -> a b

Example (direct):
    NAME anbncn
    TYPE direct
    TERMINALS a b c
    PATTERN a:n b:n c:n
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Grammar:
    name: str
    gtype: str                              # 'cfg' or 'direct'
    terminals: list
    nonterminals: list = field(default_factory=list)
    start: Optional[str] = None
    rules: dict = field(default_factory=dict)   # {nonterminal: [[sym, ...], ...]}
    pattern: Optional[str] = None


def load_grammar(filepath: str) -> Grammar:
    """Parse a grammar definition file and return a Grammar object."""
    path = Path(filepath)

    name = path.stem
    gtype = None
    terminals = []
    nonterminals = []
    start = None
    rules = {}
    pattern = None
    in_rules = False

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            if line == 'RULES':
                in_rules = True
                continue

            if in_rules:
                # e.g. "S -> a S b | a b"
                lhs, _, rhs = line.partition('->')
                lhs = lhs.strip()
                for production in rhs.split('|'):
                    symbols = production.strip().split()
                    rules.setdefault(lhs, []).append(symbols)
            else:
                key, _, value = line.partition(' ')
                value = value.strip()
                if key == 'NAME':
                    name = value
                elif key == 'TYPE':
                    gtype = value
                elif key == 'TERMINALS':
                    terminals = value.split()
                elif key == 'NONTERMINALS':
                    nonterminals = value.split()
                elif key == 'START':
                    start = value
                elif key == 'PATTERN':
                    pattern = value

    return Grammar(
        name=name,
        gtype=gtype,
        terminals=terminals,
        nonterminals=nonterminals,
        start=start,
        rules=rules,
        pattern=pattern,
    )


def generate_strings(grammar: Grammar, min_n: int = 1, max_n: int = 20) -> list:
    """Return a list of valid strings for all n in [min_n, max_n]."""
    strings = []
    for n in range(min_n, max_n + 1):
        if grammar.gtype == 'cfg':
            s = _derive_cfg(grammar.rules, set(grammar.terminals), grammar.start, n)
        elif grammar.gtype == 'direct':
            s = _apply_pattern(grammar.pattern, n)
        else:
            raise ValueError(f"Unknown grammar type: {grammar.gtype!r}")
        strings.append(s)
    return strings


def _derive_cfg(rules: dict, terminals: set, symbol: str, n: int) -> str:
    """
    Derive a string from a recursive CFG with depth parameter n.

    Handles grammars with a single recursive production and a base case, e.g.:
        S -> a S b   (recursive — contains S itself)
        S -> a b     (base case)

    n=1 applies the base case; n>1 applies the recursive rule (n-1) times.

    Non-terminal symbols found in the base case (e.g. hierarchical S -> X)
    are expanded with the same n, allowing multi-level derivations.
    """
    if symbol in terminals:
        return symbol

    prods = rules[symbol]

    recursive_prod = None
    base_prod = None
    for prod in prods:
        if symbol in prod:
            recursive_prod = prod
        else:
            base_prod = prod

    # If there's only one production (no base case), treat it as the base
    if base_prod is None:
        base_prod = prods[0]

    if n <= 1 or recursive_prod is None:
        # Base case: expand each symbol; non-terminals recurse with same n
        return ''.join(
            t if t in terminals else _derive_cfg(rules, terminals, t, n)
            for t in base_prod
        )
    else:
        # Recursive case: expand, decrementing n only when we recurse on self
        return ''.join(
            _derive_cfg(rules, terminals, symbol, n - 1) if t == symbol
            else (t if t in terminals else _derive_cfg(rules, terminals, t, n))
            for t in recursive_prod
        )


def _apply_pattern(pattern: str, n: int) -> str:
    """
    Expand a direct pattern for a given n.

    Syntax: tokens separated by spaces.
      <symbol>:n  ->  symbol repeated n times
      <symbol>    ->  symbol literal

    Example: 'a:n b:n c:n' with n=3  ->  'aaabbbccc'
    """
    parts = []
    for token in pattern.split():
        if ':n' in token:
            symbol = token.replace(':n', '')
            parts.append(symbol * n)
        else:
            parts.append(token)
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Tokenisation helpers (used by train.py and probe.py)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']


def build_vocab(grammar: Grammar) -> dict:
    """
    Build a character-level vocabulary from the grammar's terminal symbols.
    Special tokens are prepended so their IDs are stable (PAD=0, CLS=1, ...).
    """
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for t in grammar.terminals:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def tokenize(string: str, vocab: dict, add_special: bool = True) -> list:
    """
    Convert a raw string to a list of token IDs (character-level).

    If add_special=True, wraps with [CLS] ... [SEP] as Barry recommended.
    Padding is left to the collation step in train.py.
    """
    unk_id = vocab['[UNK]']
    ids = [vocab.get(c, unk_id) for c in string]
    if add_special:
        ids = [vocab['[CLS]']] + ids + [vocab['[SEP]']]
    return ids
