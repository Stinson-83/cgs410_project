"""
metrics/comparison.py — Attention Graph vs Dependency Tree Comparison

Implements four metrics from the research methodology:
    1. Dependency Edge Recovery Rate (DERR)
    2. Undirected Structural Overlap (USO)
    3. Tree Distance Correlation (TDC)
    4. Attention Mass on True Head (AMTH)

All metrics operate on word-level indices and handle edge cases
(disconnected graphs, missing edges, etc.).
"""

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from typing import List, Tuple, Dict, Any, Optional


def dependency_edge_recovery_rate(
    attention_graph: nx.DiGraph,
    gold_edges: List[Tuple[int, int]],
) -> float:
    """
    Fraction of gold dependency edges that are present in the attention graph.

    DERR = |E_gold ∩ E_attn| / |E_gold|

    Args:
        attention_graph: Directed attention graph.
        gold_edges: List of (dependent_idx, head_idx) from dependency parse.

    Returns:
        Score in [0, 1]. Higher = better syntactic alignment.
    """
    if not gold_edges:
        return 0.0

    attn_edges = set(attention_graph.edges())
    recovered = sum(1 for edge in gold_edges if edge in attn_edges)
    return recovered / len(gold_edges)


def undirected_structural_overlap(
    attention_graph: nx.DiGraph,
    gold_undirected_edges: set,
) -> float:
    """
    Jaccard similarity between undirected edge sets of attention graph
    and dependency tree.

    USO = |E_gold ∩ E_attn| / |E_gold ∪ E_attn|  (undirected)

    Args:
        attention_graph: Directed attention graph.
        gold_undirected_edges: Set of frozenset({i, j}) from dependency tree.

    Returns:
        Jaccard score in [0, 1].
    """
    attn_undirected = {frozenset([u, v]) for u, v in attention_graph.edges()}

    intersection = gold_undirected_edges & attn_undirected
    union = gold_undirected_edges | attn_undirected

    if not union:
        return 0.0

    return len(intersection) / len(union)


def tree_distance_correlation(
    attention_graph: nx.DiGraph,
    gold_adjacency: np.ndarray,
) -> float:
    """
    Spearman correlation between pairwise shortest-path distances in the
    attention graph and the gold dependency tree.

    We convert both to undirected graphs for shortest-path computation.

    Args:
        attention_graph: Directed attention graph.
        gold_adjacency: Adjacency matrix (n x n) of gold dependency tree.

    Returns:
        Spearman correlation coefficient in [-1, 1]. Higher = better.
        Returns 0 if computation is not possible.
    """
    n = gold_adjacency.shape[0]
    if n < 3:
        return 0.0

    # Build undirected gold tree graph
    gold_graph = nx.Graph()
    for i in range(n):
        gold_graph.add_node(i)
        for j in range(n):
            if gold_adjacency[i, j] > 0:
                gold_graph.add_edge(i, j)

    # Convert attention graph to undirected
    attn_undirected = attention_graph.to_undirected()

    # Compute pairwise shortest paths
    gold_dists = []
    attn_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            # Gold distance
            try:
                gd = nx.shortest_path_length(gold_graph, i, j)
            except nx.NetworkXNoPath:
                gd = n  # large penalty for disconnected nodes

            # Attention distance
            try:
                ad = nx.shortest_path_length(attn_undirected, i, j)
            except nx.NetworkXNoPath:
                ad = n

            gold_dists.append(gd)
            attn_dists.append(ad)

    if len(set(gold_dists)) < 2 or len(set(attn_dists)) < 2:
        return 0.0

    corr, _ = spearmanr(gold_dists, attn_dists)
    return float(corr) if not np.isnan(corr) else 0.0


def attention_mass_on_true_head(
    attention_matrix: np.ndarray,
    head_indices: List[int],
) -> float:
    """
    Mean attention weight each token places on its true syntactic head.

    AMTH = (1/N) * Σ_i attention[i, head(i)]

    Args:
        attention_matrix: Word-level attention matrix (n_words x n_words).
        head_indices: head_indices[i] = index of syntactic head of token i.
                      -1 for root token.

    Returns:
        Score in [0, 1]. Higher = more attention on correct head.
    """
    total = 0.0
    count = 0

    for i, head_idx in enumerate(head_indices):
        if head_idx >= 0 and head_idx < attention_matrix.shape[1]:
            total += attention_matrix[i, head_idx]
            count += 1

    return total / count if count > 0 else 0.0


def compute_all_metrics(
    attention_graph: nx.DiGraph,
    attention_matrix: np.ndarray,
    gold_edges: List[Tuple[int, int]],
    gold_undirected_edges: set,
    gold_adjacency: np.ndarray,
    head_indices: List[int],
) -> Dict[str, float]:
    """
    Compute all four metrics for one (layer, head) on one sentence.

    Returns:
        Dictionary with keys: "DERR", "USO", "TDC", "AMTH"
    """
    return {
        "DERR": dependency_edge_recovery_rate(attention_graph, gold_edges),
        "USO": undirected_structural_overlap(attention_graph, gold_undirected_edges),
        "TDC": tree_distance_correlation(attention_graph, gold_adjacency),
        "AMTH": attention_mass_on_true_head(attention_matrix, head_indices),
    }
