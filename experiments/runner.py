"""
experiments/runner.py — Full Experiment Orchestrator

Runs the complete pipeline:
    1. Generate sentences at each depth
    2. Parse gold dependency trees
    3. Extract attention from transformer
    4. Build attention graphs
    5. Compute all metrics per (depth, layer, head)
    6. Aggregate and save results

Designed for reproducibility: uses fixed seeds, logs everything, and exports
structured results for downstream analysis and plotting.
"""

import json
import os
import time
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any, Optional

from data.generator import generate_sentences
from parsing.dependency_parser import parse_sentence, get_gold_edges, get_undirected_edges
from models.attention_extractor import AttentionExtractor
from graphs.attention_graph import build_attention_graph
from metrics.comparison import compute_all_metrics

logger = logging.getLogger(__name__)


def run_experiment(
    model_name: str = "bert-base-uncased",
    max_depth: int = 7,
    num_sentences: int = 10,
    seed: int = 42,
    pruning_strategy: str = "top_k",
    top_k: int = 1,
    attention_threshold: float = 0.1,
    device: str = "cpu",
    results_dir: str = "results",
    custom_sentences: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Execute the full experimental pipeline.

    Args:
        model_name: HuggingFace model name.
        max_depth: Maximum recursion depth.
        num_sentences: Sentences per depth.
        seed: Random seed.
        pruning_strategy: "top_k" or "threshold".
        top_k: Top-k for attention graph pruning.
        attention_threshold: Threshold for attention graph pruning.
        device: "cpu" or "cuda".
        results_dir: Output directory.
        custom_sentences: A list of specific sentences to test, bypassing generation.

    Returns:
        Complete results dictionary with per-depth, per-layer, per-head metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    start_time = time.time()

    # ── Step 1: Generate or use custom sentences ─────────────────────────────
    if custom_sentences:
        logger.info(f"Using {len(custom_sentences)} custom sentences sequentially mapped to depths")
        max_depth = len(custom_sentences)
        all_sentences = {depth: [(sent, "custom")] for depth, sent in enumerate(custom_sentences, start=1)}
    else:
        logger.info(f"Generating sentences (max_depth={max_depth}, "
                    f"num_per_depth={num_sentences})...")
        all_sentences = generate_sentences(max_depth, num_sentences, seed)

    # ── Step 2: Initialize model ─────────────────────────────────────────────
    logger.info(f"Loading model: {model_name} on {device}...")
    extractor = AttentionExtractor(model_name, device)

    # ── Step 3: Run pipeline ─────────────────────────────────────────────────
    # Structure: results[depth] = list of per-sentence results
    # Each per-sentence result: metrics[layer][head] = {DERR, USO, TDC, AMTH}
    all_results: Dict[int, list] = {}
    example_data = {}  # Store one example per depth for visualization

    for depth in tqdm(range(1, max_depth + 1), desc="Depths"):
        depth_results = []

        for sent_idx, (sentence, template) in enumerate(
            tqdm(all_sentences[depth], desc=f"  Depth {depth}", leave=False)
        ):
            try:
                # ── Parse gold tree ──────────────────────────────────────────
                parse = parse_sentence(sentence)
                tokens = parse["tokens"]
                gold_edges = get_gold_edges(parse)
                gold_undirected = get_undirected_edges(parse)
                gold_adj = parse["adjacency_matrix"]
                head_indices = parse["head_indices"]

                # ── Extract attention ────────────────────────────────────────
                attn_result = extractor.extract(sentence, tokens)
                attention = attn_result["attention"]  # (layers, heads, words, words)
                n_layers, n_heads = attention.shape[:2]

                # ── Compute metrics per (layer, head) ────────────────────────
                sentence_metrics = []

                for layer in range(n_layers):
                    layer_metrics = []
                    for head in range(n_heads):
                        attn_matrix = attention[layer, head]

                        # Build attention graph
                        attn_graph = build_attention_graph(
                            attn_matrix, tokens,
                            strategy=pruning_strategy,
                            top_k=top_k,
                            threshold=attention_threshold,
                        )

                        # Compute all 4 metrics
                        metrics = compute_all_metrics(
                            attn_graph, attn_matrix,
                            gold_edges, gold_undirected,
                            gold_adj, head_indices,
                        )
                        layer_metrics.append(metrics)

                    sentence_metrics.append(layer_metrics)

                depth_results.append({
                    "sentence": sentence,
                    "template": template,
                    "n_tokens": len(tokens),
                    "tokens": tokens,
                    "metrics": sentence_metrics,  # [layer][head] → dict
                })

                # Save first sentence of each depth for visualization
                if sent_idx == 0:
                    example_data[depth] = {
                        "sentence": sentence,
                        "tokens": tokens,
                        "attention": attention,
                        "gold_adj": gold_adj,
                        "head_indices": head_indices,
                        "parse": parse,
                    }

            except Exception as e:
                logger.warning(f"Error processing '{sentence}': {e}")
                continue

        all_results[depth] = depth_results

    # ── Step 4: Aggregate metrics ────────────────────────────────────────────
    aggregated = _aggregate_metrics(all_results)

    # ── Step 5: Save results ─────────────────────────────────────────────────
    results_path = os.path.join(results_dir, "metrics_results.json")
    _save_results(aggregated, results_path)

    elapsed = time.time() - start_time
    logger.info(f"Experiment completed in {elapsed:.1f}s. "
                f"Results saved to {results_path}")

    return {
        "aggregated": aggregated,
        "raw_results": all_results,
        "example_data": example_data,
        "n_layers": n_layers if all_results else 12,
        "n_heads": n_heads if all_results else 12,
    }


def _aggregate_metrics(
    all_results: Dict[int, list],
) -> Dict[str, Any]:
    """
    Aggregate per-sentence metrics into depth × layer × head summaries.

    Returns:
        Dictionary with:
            - "per_depth_layer": {depth: {layer: {metric: {mean, std}}}}
            - "per_depth_best_layer": {depth: {metric: {mean, std, best_layer}}}
            - "per_layer_across_depths": {layer: {depth: {metric: {mean, std}}}}
    """
    metric_names = ["DERR", "USO", "TDC", "AMTH"]

    per_depth_layer = {}
    per_depth_best_layer = {}

    for depth, sentences in all_results.items():
        if not sentences:
            continue

        n_layers = len(sentences[0]["metrics"])
        n_heads = len(sentences[0]["metrics"][0])

        # Collect all metric values per (layer, head) across sentences
        # Shape: [n_sentences, n_layers, n_heads, 4]
        all_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for sent_data in sentences:
            for layer in range(n_layers):
                for head in range(n_heads):
                    for metric in metric_names:
                        val = sent_data["metrics"][layer][head][metric]
                        all_values[layer][head][metric].append(val)

        # Aggregate per (depth, layer) → average over heads and sentences
        depth_layer_agg = {}
        for layer in range(n_layers):
            layer_agg = {}
            for metric in metric_names:
                # Collect values across all heads and sentences
                values = []
                for head in range(n_heads):
                    values.extend(all_values[layer][head][metric])
                layer_agg[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            depth_layer_agg[layer] = layer_agg

        per_depth_layer[depth] = depth_layer_agg

        # Find best layer per depth (highest mean) for each metric
        best_layers = {}
        for metric in metric_names:
            best_mean = -float("inf")
            best_layer = 0
            for layer in range(n_layers):
                m = depth_layer_agg[layer][metric]["mean"]
                if m > best_mean:
                    best_mean = m
                    best_layer = layer
            best_layers[metric] = {
                "mean": best_mean,
                "std": depth_layer_agg[best_layer][metric]["std"],
                "best_layer": best_layer,
            }
        per_depth_best_layer[depth] = best_layers

    return {
        "per_depth_layer": per_depth_layer,
        "per_depth_best_layer": per_depth_best_layer,
    }


def _save_results(aggregated: Dict, path: str):
    """Save aggregated results as JSON (converting int keys to strings)."""
    def _convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): _convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_keys(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(path, "w") as f:
        json.dump(_convert_keys(aggregated), f, indent=2)


def print_observations(aggregated: Dict):
    """
    Print key experimental observations to stdout.
    Summarizes depth-vs-alignment trends and identifies best layers.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTAL OBSERVATIONS")
    print("=" * 70)

    per_depth_best = aggregated.get("per_depth_best_layer", {})
    metric_names = ["DERR", "USO", "TDC", "AMTH"]
    metric_descriptions = {
        "DERR": "Dependency Edge Recovery Rate",
        "USO": "Undirected Structural Overlap",
        "TDC": "Tree Distance Correlation",
        "AMTH": "Attention Mass on True Head",
    }

    # ── Overall trends ───────────────────────────────────────────────────
    print("\n1. DEPTH vs ALIGNMENT (best-layer scores)")
    print("-" * 50)

    for metric in metric_names:
        print(f"\n  {metric} ({metric_descriptions[metric]}):")
        depths = sorted(per_depth_best.keys(), key=int)

        values = []
        for d in depths:
            info = per_depth_best[d][metric]
            mean_val = info["mean"]
            std_val = info["std"]
            best_layer = info["best_layer"]
            values.append(mean_val)
            print(f"    Depth {d}: {mean_val:.4f} ± {std_val:.4f} "
                  f"(best layer: {best_layer})")

        # Trend analysis
        if len(values) >= 2:
            if values[-1] < values[0]:
                decline = (values[0] - values[-1]) / max(values[0], 1e-8) * 100
                print(f"    → DECLINE: {decline:.1f}% from depth {depths[0]} "
                      f"to {depths[-1]}")
            else:
                print(f"    → STABLE/INCREASING across depths")

    # ── Best syntactic layers ────────────────────────────────────────────
    print("\n2. BEST SYNTACTIC LAYERS")
    print("-" * 50)

    per_depth_layer = aggregated.get("per_depth_layer", {})
    if per_depth_layer:
        first_depth = sorted(per_depth_layer.keys(), key=int)[0]
        layers_data = per_depth_layer[first_depth]
        for metric in metric_names:
            best_layer = max(
                layers_data.keys(),
                key=lambda l: layers_data[l][metric]["mean"]
            )
            val = layers_data[best_layer][metric]["mean"]
            print(f"  {metric}: Layer {best_layer} "
                  f"(score={val:.4f}) at depth {first_depth}")

    # ── Conclusion ───────────────────────────────────────────────────────
    print("\n3. CONCLUSIONS")
    print("-" * 50)

    derr_values = [per_depth_best[d]["DERR"]["mean"]
                   for d in sorted(per_depth_best.keys(), key=int)]
    if len(derr_values) >= 2 and derr_values[-1] < derr_values[0] * 0.8:
        print("  ⚠ Attention-syntax alignment DEGRADES significantly "
              "with recursion depth.")
        print("  → Supports hypothesis: transformers rely more on local "
              "token interactions at higher depths.")
    else:
        print("  ✓ Attention-syntax alignment remains relatively stable "
              "across depths.")
        print("  → Transformers may maintain hierarchical structure better "
              "than hypothesized.")

    print("\n" + "=" * 70)
