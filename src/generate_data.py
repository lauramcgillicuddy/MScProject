"""
Generate training data from a grammar definition file.

Usage:
    python generate_data.py grammars/anbn.txt
    python generate_data.py grammars/anbn.txt --min-n 1 --max-n 100
    python generate_data.py grammars/anbncn.txt --min-n 1 --max-n 50 --output data/anbncn.txt

Output format: one string per line, e.g.
    ab
    aabb
    aaabbb
    ...

Run this script for each grammar file before training.
"""

import argparse
from pathlib import Path

from grammar_loader import load_grammar, generate_strings


def main():
    parser = argparse.ArgumentParser(description='Generate grammar strings for model training.')
    parser.add_argument('grammar', help='Path to grammar definition file (e.g. grammars/anbn.txt)')
    parser.add_argument('--min-n', type=int, default=1, help='Minimum n (default: 1)')
    parser.add_argument('--max-n', type=int, default=20, help='Maximum n (default: 20)')
    parser.add_argument('--output', default=None, help='Output file path (default: data/<name>_n<min>-<max>.txt)')
    args = parser.parse_args()

    grammar = load_grammar(args.grammar)
    strings = generate_strings(grammar, min_n=args.min_n, max_n=args.max_n)

    output_path = args.output or f'data/{grammar.name}_n{args.min_n}-{args.max_n}.txt'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for s in strings:
            f.write(s + '\n')

    print(f"Grammar : {grammar.name}  ({grammar.gtype})")
    print(f"Strings : {len(strings)}  (n={args.min_n}..{args.max_n})")
    print(f"Output  : {output_path}")
    print(f"Lengths : {min(len(s) for s in strings)} .. {max(len(s) for s in strings)} chars")


if __name__ == '__main__':
    main()
