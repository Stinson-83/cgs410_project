"""
graphs/attention_graph.py — Attention Matrix → Directed Graph Conversion

Converts word-level attention matrices into pruned directed graphs using
either top-k or threshold-based filtering strategies.
"""

import numpy as np
import networkx as nx
from typing import List, Optional


def build_attention_graph(
    attention_matrix: np.ndarray,
    tokens: List[str],
    strategy: str = "top_k",
    top_k: int = 1,
    threshold: float = 0.1,
) -> nx.DiGraph:
    """
    Convert a single attention matrix (one layer, one head) into a directed graph.

    Each token becomes a node. Edges represent attention from one token to another,
    pruned according to the chosen strategy.

    Args:
        attention_matrix: Shape (n_words, n_words). Entry [i, j] = attention from
                          token i to token j.
        tokens: List of word strings (used as node labels).
        strategy: "top_k" or "threshold".
        top_k: Number of edges to keep per source token (if strategy="top_k").
        threshold: Minimum attention weight to keep (if strategy="threshold").

    Returns:
        networkx.DiGraph with nodes labeled by token strings and edges weighted
        by attention values.
    """
    n = len(tokens)
    G = nx.DiGraph()

    # Add all nodes
    for i, tok in enumerate(tokens):
        G.add_node(i, label=tok)

    if strategy == "top_k":
        for i in range(n):
            row = attention_matrix[i]
            # Get indices of top-k attended tokens
            if len(row) <= top_k:
                top_indices = list(range(len(row)))
            else:
                top_indices = np.argsort(row)[-top_k:]
            for j in top_indices:
                if row[j] > 0:
                    G.add_edge(i, j, weight=float(row[j]))

    elif strategy == "threshold":
        for i in range(n):
            for j in range(n):
                if attention_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=float(attention_matrix[i, j]))

    elif strategy == "mst":
        dense_G = nx.DiGraph()
        for i in range(n):
            dense_G.add_node(i, label=tokens[i])
        for i in range(n):
            for j in range(n):
                if i != j:
                    # dense_G requires head -> dependent so that dependent has exactly 1 incoming edge
                    dense_G.add_edge(j, i, weight=float(attention_matrix[i, j]))
        try:
            mst = nx.maximum_spanning_arborescence(dense_G, preserve_attrs=True)
            for head, dependent, data in mst.edges(data=True):
                # The normal pipeline expects dependent -> head
                G.add_edge(dependent, head, weight=data.get("weight", float(attention_matrix[dependent, head])))
        except nx.NetworkXException:
            # Fallback to top-k if parsing fails
            return build_attention_graph(attention_matrix, tokens, strategy="top_k", top_k=1)

    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")

    return G


def batch_build_graphs(
    attention_all: np.ndarray,
    tokens: List[str],
    strategy: str = "top_k",
    top_k: int = 1,
    threshold: float = 0.1,
) -> List[List[nx.DiGraph]]:
    """
    Build attention graphs for all layers and heads.

    Args:
        attention_all: Shape (n_layers, n_heads, n_words, n_words).
        tokens: Word tokens.
        strategy: Pruning strategy.
        top_k: Top-k parameter.
        threshold: Threshold parameter.

    Returns:
        Nested list: graphs[layer][head] = nx.DiGraph
    """
    n_layers, n_heads = attention_all.shape[:2]
    graphs = []

    for layer in range(n_layers):
        layer_graphs = []
        for head in range(n_heads):
            G = build_attention_graph(
                attention_all[layer, head],
                tokens,
                strategy=strategy,
                top_k=top_k,
                threshold=threshold,
            )
            layer_graphs.append(G)
        graphs.append(layer_graphs)

    return graphs


def get_attention_edges(G: nx.DiGraph) -> List[tuple]:
    """Extract directed edge list from attention graph."""
    return list(G.edges())


def get_undirected_attention_edges(G: nx.DiGraph) -> set:
    """Extract undirected edge set from attention graph."""
    return {frozenset([u, v]) for u, v in G.edges()}
