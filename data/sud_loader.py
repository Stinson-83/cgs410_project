"""
data/sud_loader.py — SUD Treebank CoNLL-U Loader

Loads sentences from Surface-syntactic Universal Dependencies (SUD) treebanks,
computes dependency tree depth, bins sentences by depth, and performs stratified
sampling for the experimental pipeline.

Supported treebanks:
    - SUD_English-EWT
    - SUD_German-GSD
    - SUD_Hindi-HDTB

References:
    Gerdes, K., et al. (2018). SUD or Surface-Syntactic Universal Dependencies. UDW, EMNLP.
"""

import os
import random
import logging
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_conllu_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse a CoNLL-U file into a list of sentence dictionaries.

    Each sentence dict contains:
        - "tokens": list of word strings
        - "heads": list of head indices (0-indexed, -1 for root)
        - "deps": list of (token_idx, head_idx, dep_label) triples
        - "adjacency_matrix": numpy array (n x n)
        - "head_indices": list of head indices per token
        - "sentence_text": space-joined tokens

    Args:
        filepath: Path to a .conllu file.

    Returns:
        List of parsed sentence dictionaries.
    """
    sentences = []
    current_tokens = []
    current_heads = []
    current_deps = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            if not line:
                # End of sentence
                if current_tokens:
                    sent = _build_sentence_dict(
                        current_tokens, current_heads, current_deps
                    )
                    if sent is not None:
                        sentences.append(sent)
                current_tokens = []
                current_heads = []
                current_deps = []
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue

            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue

            word = fields[1]
            head = fields[6]
            deprel = fields[7]

            try:
                head_idx = int(head)
            except ValueError:
                continue

            current_tokens.append(word)
            current_heads.append(head_idx)
            current_deps.append(deprel)

    # Handle last sentence if file doesn't end with blank line
    if current_tokens:
        sent = _build_sentence_dict(current_tokens, current_heads, current_deps)
        if sent is not None:
            sentences.append(sent)

    return sentences


def _build_sentence_dict(
    tokens: List[str],
    heads: List[int],
    deps: List[str],
) -> Optional[Dict[str, Any]]:
    """Build a standardised sentence dict from raw CoNLL-U fields."""
    n = len(tokens)
    if n < 2:
        return None

    # Convert from 1-indexed (CoNLL-U) to 0-indexed; root's head (0) becomes -1
    head_indices = []
    dep_triples = []
    adjacency_matrix = np.zeros((n, n), dtype=np.float32)

    for i, (head, dep) in enumerate(zip(heads, deps)):
        if head == 0:
            h_idx = -1  # root
        else:
            h_idx = head - 1  # 0-indexed

        head_indices.append(h_idx)
        dep_triples.append((i, h_idx, dep))

        if h_idx >= 0 and h_idx < n:
            adjacency_matrix[i, h_idx] = 1.0

    return {
        "tokens": tokens,
        "heads": head_indices,
        "deps": dep_triples,
        "adjacency_matrix": adjacency_matrix,
        "head_indices": head_indices,
        "sentence_text": " ".join(tokens),
    }


def compute_tree_depth(head_indices: List[int]) -> int:
    """
    Compute the maximum depth of a dependency tree.

    Depth is defined as the maximum distance from any leaf to the root,
    following Karlsson (2007).

    Args:
        head_indices: 0-indexed head indices (-1 for root).

    Returns:
        Maximum tree depth (1-indexed: a single-word sentence has depth 1).
    """
    n = len(head_indices)
    if n == 0:
        return 0

    depths = [0] * n
    computed = [False] * n

    def _get_depth(i: int, visited: set) -> int:
        if computed[i]:
            return depths[i]
        if i in visited:
            return 1  # cycle guard
        visited.add(i)

        head = head_indices[i]
        if head < 0 or head >= n:
            # Root node
            depths[i] = 1
            computed[i] = True
            return 1

        depths[i] = _get_depth(head, visited) + 1
        computed[i] = True
        return depths[i]

    for i in range(n):
        if not computed[i]:
            _get_depth(i, set())

    return max(depths) if depths else 0


def bin_sentences_by_depth(
    sentences: List[Dict[str, Any]],
    max_depth: int = 7,
    min_tokens: int = 5,
    max_tokens: int = 25,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Bin sentences by dependency tree depth with length filtering.

    Args:
        sentences: List of parsed sentence dicts.
        max_depth: Maximum depth bin (inclusive).
        min_tokens: Minimum token count (inclusive).
        max_tokens: Maximum token count (inclusive).

    Returns:
        Dictionary mapping depth -> list of sentence dicts.
    """
    bins: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for sent in sentences:
        n = len(sent["tokens"])
        if n < min_tokens or n > max_tokens:
            continue

        depth = compute_tree_depth(sent["head_indices"])

        if 1 <= depth <= max_depth:
            bins[depth].append(sent)

    return dict(bins)


def stratified_sample(
    binned_sentences: Dict[int, List[Dict[str, Any]]],
    samples_per_depth: int = 30,
    max_depth: int = 7,
    seed: int = 42,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Stratified random sample of sentences from depth bins.

    If a bin has fewer than `samples_per_depth` sentences,
    samples with replacement.

    Args:
        binned_sentences: Output of bin_sentences_by_depth().
        samples_per_depth: Number of sentences per depth.
        max_depth: Maximum depth bin.
        seed: Random seed.

    Returns:
        Dictionary mapping depth -> sampled list of sentence dicts.
    """
    rng = random.Random(seed)
    sampled: Dict[int, List[Dict[str, Any]]] = {}

    for depth in range(1, max_depth + 1):
        available = binned_sentences.get(depth, [])

        if not available:
            logger.warning(
                f"No sentences available at depth {depth}; skipping."
            )
            continue

        if len(available) >= samples_per_depth:
            sampled[depth] = rng.sample(available, samples_per_depth)
        else:
            # Sample with replacement
            logger.info(
                f"Depth {depth}: only {len(available)} sentences, "
                f"sampling with replacement to get {samples_per_depth}."
            )
            sampled[depth] = rng.choices(available, k=samples_per_depth)

    return sampled


def load_sud_treebank(
    conllu_path: str,
    max_depth: int = 7,
    samples_per_depth: int = 30,
    min_tokens: int = 5,
    max_tokens: int = 25,
    seed: int = 42,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    End-to-end loader: parse CoNLL-U -> bin by depth -> stratified sample.

    Args:
        conllu_path: Path to the .conllu file.
        max_depth: Maximum depth bin.
        samples_per_depth: Sentences to sample per depth.
        min_tokens: Minimum token count.
        max_tokens: Maximum token count.
        seed: Random seed.

    Returns:
        Dictionary mapping depth -> list of sampled sentence dicts.
    """
    logger.info(f"Loading SUD treebank from: {conllu_path}")
    sentences = parse_conllu_file(conllu_path)
    logger.info(f"  Parsed {len(sentences)} sentences.")

    binned = bin_sentences_by_depth(sentences, max_depth, min_tokens, max_tokens)
    logger.info(
        f"  Depth distribution: "
        + ", ".join(f"d{d}={len(binned.get(d, []))}" for d in range(1, max_depth + 1))
    )

    sampled = stratified_sample(binned, samples_per_depth, max_depth, seed)
    logger.info(
        f"  Sampled {sum(len(v) for v in sampled.values())} sentences "
        f"across {len(sampled)} depth bins."
    )

    return sampled


def get_treebank_stats(
    conllu_path: str,
    max_depth: int = 7,
    min_tokens: int = 5,
    max_tokens: int = 25,
) -> Dict[int, int]:
    """
    Get depth distribution counts without sampling (for reporting).

    Returns:
        Dictionary mapping depth -> count of filtered sentences.
    """
    sentences = parse_conllu_file(conllu_path)
    binned = bin_sentences_by_depth(sentences, max_depth, min_tokens, max_tokens)
    return {d: len(binned.get(d, [])) for d in range(1, max_depth + 1)}


# ─── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m data.sud_loader <path/to/file.conllu>")
        sys.exit(1)

    path = sys.argv[1]
    stats = get_treebank_stats(path)
    print(f"\nDepth distribution for: {path}")
    for d, count in sorted(stats.items()):
        print(f"  Depth {d}: {count} sentences")

    sampled = load_sud_treebank(path, samples_per_depth=5)
    for d, sents in sorted(sampled.items()):
        print(f"\n--- Depth {d} (sampled {len(sents)}) ---")
        for s in sents[:2]:
            print(f"  {s['sentence_text']}")
