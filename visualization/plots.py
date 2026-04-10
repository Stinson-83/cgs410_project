"""
visualization/plots.py — Publication-Quality Visualization Suite

Generates:
    1. Depth vs alignment score (line plot per metric)
    2. Layer vs alignment heatmap (layer × depth)
    3. Attention heatmap for a single sentence (with gold edges highlighted)
    4. Dependency tree overlay (gold tree + attention graph side-by-side)
    5. Head-level analysis (best heads per layer)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from typing import Dict, Any, List, Optional

# ─── Plotting defaults ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})


def plot_depth_vs_alignment(
    aggregated: Dict,
    output_dir: str,
    filename: str = "depth_vs_alignment.png",
):
    """
    Line plot of depth (x) vs alignment score (y) for each metric.
    Uses the best-layer score for each depth.
    """
    per_depth_best = aggregated.get("per_depth_best_layer", {})
    if not per_depth_best:
        return

    metric_names = ["DERR", "USO", "TDC", "AMTH"]
    metric_labels = {
        "DERR": "Edge Recovery Rate",
        "USO": "Structural Overlap",
        "TDC": "Tree Distance Corr.",
        "AMTH": "Attn Mass on Head",
    }
    colors = {"DERR": "#2196F3", "USO": "#4CAF50", "TDC": "#FF9800", "AMTH": "#E91E63"}
    markers = {"DERR": "o", "USO": "s", "TDC": "^", "AMTH": "D"}

    fig, ax = plt.subplots(figsize=(10, 6))

    depths = sorted(per_depth_best.keys(), key=int)

    for metric in metric_names:
        means = [per_depth_best[d][metric]["mean"] for d in depths]
        stds = [per_depth_best[d][metric]["std"] for d in depths]

        ax.errorbar(
            [int(d) for d in depths], means, yerr=stds,
            label=metric_labels[metric],
            marker=markers[metric], color=colors[metric],
            capsize=4, linewidth=2, markersize=7, alpha=0.9,
        )

    ax.set_xlabel("Recursion Depth", fontsize=12)
    ax.set_ylabel("Alignment Score", fontsize=12)
    ax.set_title("Attention–Syntax Alignment vs Recursion Depth\n(Best Layer per Depth)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([int(d) for d in depths])

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_layer_vs_alignment_heatmap(
    aggregated: Dict,
    output_dir: str,
    filename: str = "layer_vs_alignment_heatmap.png",
):
    """
    Heatmap of (layer × depth) for each metric.
    4 subplots in a 2×2 grid.
    """
    per_depth_layer = aggregated.get("per_depth_layer", {})
    if not per_depth_layer:
        return

    metric_names = ["DERR", "USO", "TDC", "AMTH"]
    metric_labels = {
        "DERR": "Edge Recovery Rate",
        "USO": "Structural Overlap",
        "TDC": "Tree Distance Corr.",
        "AMTH": "Attn Mass on Head",
    }

    depths = sorted(per_depth_layer.keys(), key=int)
    n_layers = len(per_depth_layer[depths[0]])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, metric in enumerate(metric_names):
        ax = axes[idx // 2][idx % 2]

        # Build matrix: rows=layers, cols=depths
        matrix = np.zeros((n_layers, len(depths)))
        for col, d in enumerate(depths):
            for layer in range(n_layers):
                matrix[layer, col] = per_depth_layer[d][layer][metric]["mean"]

        sns.heatmap(
            matrix, ax=ax,
            xticklabels=[str(d) for d in depths],
            yticklabels=range(n_layers),
            cmap="YlOrRd" if metric != "TDC" else "RdYlGn",
            annot=True if n_layers <= 12 and len(depths) <= 10 else False,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Recursion Depth")
        ax.set_ylabel("Layer")
        ax.set_title(metric_labels[metric], fontweight="bold")

    fig.suptitle("Attention–Syntax Alignment: Layer × Depth",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    gold_adj: np.ndarray,
    layer: int,
    head: int,
    output_dir: str,
    filename: str = "attention_heatmap_example.png",
    sentence: str = "",
):
    """
    Attention heatmap for a single (layer, head) with gold dependency edges
    marked as red squares.
    """
    n = len(tokens)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))

    # Plot attention weights
    sns.heatmap(
        attention_matrix, ax=ax,
        xticklabels=tokens, yticklabels=tokens,
        cmap="Blues", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Attention Weight"},
    )

    # Overlay gold dependency edges as red rectangles
    for i in range(n):
        for j in range(n):
            if gold_adj[i, j] > 0:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.5,
                ))

    ax.set_xlabel("Attended To (Key)", fontsize=11)
    ax.set_ylabel("Attending From (Query)", fontsize=11)
    title = f"Attention Heatmap (Layer {layer}, Head {head})"
    if sentence:
        title += f"\n\"{sentence}\""
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Legend
    gold_patch = mpatches.Patch(
        facecolor="none", edgecolor="red", linewidth=2,
        label="Gold dependency edge"
    )
    ax.legend(handles=[gold_patch], loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_dependency_overlay(
    attention_matrix: np.ndarray,
    tokens: List[str],
    gold_adj: np.ndarray,
    head_indices: List[int],
    layer: int,
    head: int,
    output_dir: str,
    filename: str = "dependency_overlay_example.png",
    sentence: str = "",
    top_k: int = 1,
):
    """
    Side-by-side graph visualization:
        Left: Gold dependency tree
        Right: Attention graph (top-k edges)
    """
    n = len(tokens)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, n * 0.4)))

    # ── Gold dependency tree ─────────────────────────────────────────────
    gold_G = nx.DiGraph()
    for i, tok in enumerate(tokens):
        gold_G.add_node(i, label=tok)
    for i, head_idx in enumerate(head_indices):
        if head_idx >= 0:
            gold_G.add_edge(i, head_idx)

    # Layout: linear (sentence order)
    pos = {i: (i, 0) for i in range(n)}
    labels = {i: tok for i, tok in enumerate(tokens)}

    nx.draw_networkx_nodes(gold_G, pos, ax=ax1, node_color="#4CAF50",
                           node_size=600, alpha=0.8)
    nx.draw_networkx_labels(gold_G, pos, labels, ax=ax1, font_size=8)
    nx.draw_networkx_edges(gold_G, pos, ax=ax1, edge_color="#333",
                           arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.2",
                           width=1.5)
    ax1.set_title("Gold Dependency Tree", fontweight="bold", fontsize=11)
    ax1.axis("off")

    # ── Attention graph ──────────────────────────────────────────────────
    attn_G = nx.DiGraph()
    for i, tok in enumerate(tokens):
        attn_G.add_node(i, label=tok)

    for i in range(n):
        row = attention_matrix[i]
        top_indices = np.argsort(row)[-top_k:]
        for j in top_indices:
            if row[j] > 0:
                attn_G.add_edge(i, j, weight=float(row[j]))

    # Color edges: green if matches gold, red if not
    edge_colors = []
    gold_edge_set = set()
    for i, head_idx in enumerate(head_indices):
        if head_idx >= 0:
            gold_edge_set.add((i, head_idx))

    for u, v in attn_G.edges():
        if (u, v) in gold_edge_set:
            edge_colors.append("#4CAF50")  # match
        else:
            edge_colors.append("#F44336")  # mismatch

    nx.draw_networkx_nodes(attn_G, pos, ax=ax2, node_color="#2196F3",
                           node_size=600, alpha=0.8)
    nx.draw_networkx_labels(attn_G, pos, labels, ax=ax2, font_size=8)
    if attn_G.edges():
        nx.draw_networkx_edges(attn_G, pos, ax=ax2, edge_color=edge_colors,
                               arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.2", width=1.5)
    ax2.set_title(f"Attention Graph (L{layer}, H{head})", fontweight="bold",
                  fontsize=11)
    ax2.axis("off")

    # Legend
    match_patch = mpatches.Patch(color="#4CAF50", label="Matches gold")
    miss_patch = mpatches.Patch(color="#F44336", label="Mismatches gold")
    fig.legend(handles=[match_patch, miss_patch], loc="lower center",
               ncol=2, fontsize=10, framealpha=0.9)

    suptitle = "Dependency Structure: Gold vs Attention"
    if sentence:
        suptitle += f"\n\"{sentence}\""
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_head_analysis(
    aggregated: Dict,
    raw_results: Dict,
    output_dir: str,
    filename: str = "head_analysis.png",
):
    """
    Bar chart showing which heads per layer best capture dependency structure.
    Uses DERR metric at depth=1.
    """
    depth_key = sorted(raw_results.keys(), key=int)[0]
    sentences = raw_results[depth_key]
    if not sentences:
        return

    n_layers = len(sentences[0]["metrics"])
    n_heads = len(sentences[0]["metrics"][0])

    # Average DERR per (layer, head) at lowest depth
    avg_derr = np.zeros((n_layers, n_heads))

    for sent in sentences:
        for layer in range(n_layers):
            for head in range(n_heads):
                avg_derr[layer, head] += sent["metrics"][layer][head]["DERR"]
    avg_derr /= len(sentences)

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        avg_derr, ax=ax,
        xticklabels=range(n_heads),
        yticklabels=range(n_layers),
        cmap="YlOrRd", annot=True, fmt=".2f",
        cbar_kws={"label": "Edge Recovery Rate"},
    )
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(f"Edge Recovery Rate per (Layer, Head) at Depth {depth_key}",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(
    experiment_results: Dict[str, Any],
    output_dir: str,
):
    """
    Generate all visualization plots from experiment results.

    Args:
        experiment_results: Output of experiments.runner.run_experiment().
        output_dir: Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    aggregated = experiment_results["aggregated"]
    raw_results = experiment_results["raw_results"]
    example_data = experiment_results["example_data"]
    n_layers = experiment_results.get("n_layers", 12)
    n_heads = experiment_results.get("n_heads", 12)

    print("\nGenerating visualizations...")

    # 1. Depth vs alignment
    plot_depth_vs_alignment(aggregated, output_dir)

    # 2. Layer vs alignment heatmap
    plot_layer_vs_alignment_heatmap(aggregated, output_dir)

    # 3 & 4. Example attention heatmap and dependency overlay
    # Pick depth 1 and a middle depth for comparison
    example_depths = sorted(example_data.keys(), key=int)
    if example_depths:
        # Find best layer for DERR at depth 1
        per_depth_best = aggregated.get("per_depth_best_layer", {})
        d1 = example_depths[0]
        best_layer = 0
        if per_depth_best and d1 in per_depth_best:
            best_layer = per_depth_best[d1]["DERR"].get("best_layer", 0)

        ex = example_data[d1]
        best_head = 0

        # Find best head in best layer for this example
        if ex["attention"].shape[1] > 0:
            derr_per_head = []
            for h in range(ex["attention"].shape[1]):
                from graphs.attention_graph import build_attention_graph
                from metrics.comparison import dependency_edge_recovery_rate
                from parsing.dependency_parser import get_gold_edges

                attn_g = build_attention_graph(
                    ex["attention"][best_layer, h], ex["tokens"],
                    strategy="top_k", top_k=1,
                )
                gold_e = get_gold_edges(ex["parse"])
                derr = dependency_edge_recovery_rate(attn_g, gold_e)
                derr_per_head.append(derr)
            best_head = int(np.argmax(derr_per_head))

        # Attention heatmap
        plot_attention_heatmap(
            ex["attention"][best_layer, best_head],
            ex["tokens"], ex["gold_adj"],
            layer=best_layer, head=best_head,
            output_dir=output_dir,
            sentence=ex["sentence"],
        )

        # Dependency overlay
        plot_dependency_overlay(
            ex["attention"][best_layer, best_head],
            ex["tokens"], ex["gold_adj"], ex["head_indices"],
            layer=best_layer, head=best_head,
            output_dir=output_dir,
            sentence=ex["sentence"],
        )

        # If we have a higher depth example, also create overlay for comparison
        if len(example_depths) >= 3:
            mid_d = example_depths[len(example_depths) // 2]
            ex_mid = example_data[mid_d]
            plot_dependency_overlay(
                ex_mid["attention"][best_layer, min(best_head, ex_mid["attention"].shape[1]-1)],
                ex_mid["tokens"], ex_mid["gold_adj"], ex_mid["head_indices"],
                layer=best_layer, head=min(best_head, ex_mid["attention"].shape[1]-1),
                output_dir=output_dir,
                filename=f"dependency_overlay_depth{mid_d}.png",
                sentence=ex_mid["sentence"],
            )

    # 5. Head analysis
    plot_head_analysis(aggregated, raw_results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")
