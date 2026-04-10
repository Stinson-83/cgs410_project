"""
parsing/dependency_parser.py — Gold Dependency Tree Extraction

Uses spaCy to parse sentences into gold-standard dependency trees, represented
as both structured token lists and adjacency matrices.
"""

import numpy as np
import spacy
from typing import List, Tuple, Optional, Dict, Any

# Module-level spaCy model (lazy-loaded)
_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    """Lazy-load spaCy model to avoid overhead on import."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def parse_sentence(sentence: str) -> Dict[str, Any]:
    """
    Parse a sentence and extract its gold dependency tree.

    Args:
        sentence: Input sentence string.

    Returns:
        Dictionary with:
            - "tokens": list of token strings (words only, no punct filtering)
            - "deps": list of (token_idx, head_idx, dep_label) triples
            - "adjacency_matrix": numpy array of shape (n, n), adj[i][j] = 1
              means token j is the head of token i
            - "head_indices": list where head_indices[i] = index of head of token i
              (-1 for root)
    """
    nlp = _get_nlp()
    doc = nlp(sentence)

    tokens = [tok.text for tok in doc]
    n = len(tokens)

    # Build dependency information
    deps: List[Tuple[int, int, str]] = []
    head_indices: List[int] = []
    adjacency_matrix = np.zeros((n, n), dtype=np.float32)

    for tok in doc:
        if tok.head == tok:
            # Root token: head points to itself in spaCy
            head_idx = -1
        else:
            head_idx = tok.head.i

        deps.append((tok.i, head_idx, tok.dep_))
        head_indices.append(head_idx)

        # Directed edge: token → head (i.e., adjacency_matrix[tok_i, head_i] = 1)
        if head_idx >= 0:
            adjacency_matrix[tok.i, head_idx] = 1.0

    return {
        "tokens": tokens,
        "deps": deps,
        "adjacency_matrix": adjacency_matrix,
        "head_indices": head_indices,
    }


def get_gold_edges(parse_result: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Extract the set of directed dependency edges (dependent → head).

    Args:
        parse_result: Output of parse_sentence().

    Returns:
        List of (dependent_idx, head_idx) tuples, excluding root self-loops.
    """
    edges = []
    for tok_idx, head_idx, _ in parse_result["deps"]:
        if head_idx >= 0:
            edges.append((tok_idx, head_idx))
    return edges


def get_undirected_edges(parse_result: Dict[str, Any]) -> set:
    """
    Get undirected edge set from dependency tree.

    Returns:
        Set of frozensets {i, j} for each dependency edge.
    """
    edges = set()
    for tok_idx, head_idx, _ in parse_result["deps"]:
        if head_idx >= 0:
            edges.add(frozenset([tok_idx, head_idx]))
    return edges


# ─── CLI test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        "The dog barked.",
        "The dog that chased the cat barked.",
        "The dog that chased the cat that ate the mouse barked.",
    ]
    for sent in test_sentences:
        result = parse_sentence(sent)
        print(f"\nSentence: {sent}")
        print(f"Tokens: {result['tokens']}")
        for tok_i, head_i, label in result["deps"]:
            head_tok = result["tokens"][head_i] if head_i >= 0 else "ROOT"
            print(f"  {result['tokens'][tok_i]} --[{label}]--> {head_tok}")
