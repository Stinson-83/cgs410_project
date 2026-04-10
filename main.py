"""
main.py — CLI Entry Point for Transformer Attention Syntactic Dependency Pipeline

Usage:
    python main.py                           # Run with default settings
    python main.py --max_depth 5             # Test up to depth 5
    python main.py --model_name bert-base-uncased --device cuda
    python main.py --num_sentences 20        # More sentences per depth

This script orchestrates the full experimental pipeline:
    1. Generate controlled recursive sentences
    2. Parse gold dependency trees
    3. Extract attention weights from transformer
    4. Build attention graphs
    5. Compute alignment metrics
    6. Generate plots and observations
"""

import argparse
import logging
import os
import random
import numpy as np
import torch

import config
from experiments.runner import run_experiment, print_observations
from visualization.plots import generate_all_plots


def setup_logging():
    """Configure logging for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command-line arguments with defaults from config."""
    parser = argparse.ArgumentParser(
        description="Transformer Attention & Syntactic Dependency Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # Default settings
    python main.py --max_depth 5 --num_sentences 5    # Quick test run
    python main.py --model_name bert-base-uncased     # Specify model
    python main.py --device cuda                      # Use GPU
    python main.py --custom_sentences "The cat sat on the mat." "Another sentence."
    python main.py --pruning threshold --threshold 0.1
        """,
    )

    parser.add_argument(
        "--model_name", type=str, default=config.MODEL_NAME,
        help=f"HuggingFace model name (default: {config.MODEL_NAME})",
    )
    parser.add_argument(
        "--custom_sentences", nargs='+', default=None,
        help="Run analysis on one or more custom sentences instead of generated sentences.",
    )
    parser.add_argument(
        "--max_depth", type=int, default=config.MAX_DEPTH,
        help=f"Maximum recursion depth (default: {config.MAX_DEPTH})",
    )
    parser.add_argument(
        "--num_sentences", type=int, default=config.NUM_SENTENCES_PER_DEPTH,
        help=f"Sentences per depth level (default: {config.NUM_SENTENCES_PER_DEPTH})",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help=f"Random seed (default: {config.RANDOM_SEED})",
    )
    parser.add_argument(
        "--device", type=str, default=config.DEVICE,
        choices=["cpu", "cuda"],
        help=f"Compute device (default: {config.DEVICE})",
    )
    parser.add_argument(
        "--pruning", type=str, default="both",
        choices=["top_k", "threshold", "mst", "both"],
        help=f"Attention graph pruning strategy (default: both)",
    )
    parser.add_argument(
        "--top_k", type=int, default=config.TOP_K,
        help=f"Top-k edges per token (default: {config.TOP_K})",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.ATTENTION_THRESHOLD,
        help=f"Attention threshold (default: {config.ATTENTION_THRESHOLD})",
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.RESULTS_DIR,
        help=f"Results output directory (default: {config.RESULTS_DIR})",
    )
    parser.add_argument(
        "--figures_dir", type=str, default=config.FIGURES_DIR,
        help=f"Figures output directory (default: {config.FIGURES_DIR})",
    )

    return parser.parse_args()


def main():
    """Run the full experimental pipeline."""
    args = parse_args()

    setup_logging()
    set_seed(args.seed)

    logger = logging.getLogger("main")
    
    strategies_to_run = ["mst", "top_k"] if args.pruning == "both" else [args.pruning]
    base_results_dir = args.results_dir

    for strategy in strategies_to_run:
        # Separate outputs by pruning strategy to prevent overwriting
        current_results_dir = os.path.join(base_results_dir, strategy)
        current_figures_dir = os.path.join(current_results_dir, "figures")

        # ── Print configuration ──────────────────────────────────────────────
        print("=" * 70)
        print(f"TRANSFORMER ATTENTION & SYNTACTIC DEPENDENCY ANALYSIS ({strategy.upper()})")
        print("=" * 70)
        print(f"  Model:          {args.model_name}")
        print(f"  Max depth:      {args.max_depth}")
        print(f"  Sentences/depth: {args.num_sentences}")
        print(f"  Pruning:        {strategy} (top_k={args.top_k}, "
              f"threshold={args.threshold})")
        print(f"  Device:         {args.device}")
        print(f"  Seed:           {args.seed}")
        print(f"  Results dir:    {current_results_dir}")
        print(f"  Figures dir:    {current_figures_dir}")
        print("=" * 70)

        # ── Run experiment ───────────────────────────────────────────────────
        logger.info(f"Starting experiment for pruning strategy: {strategy}...")

        results = run_experiment(
            model_name=args.model_name,
            max_depth=args.max_depth,
            num_sentences=args.num_sentences,
            seed=args.seed,
            pruning_strategy=strategy,
            top_k=args.top_k,
            attention_threshold=args.threshold,
            device=args.device,
            results_dir=current_results_dir,
            custom_sentences=args.custom_sentences,
        )

        # ── Generate visualizations ──────────────────────────────────────────
        logger.info(f"Generating plots for {strategy}...")
        generate_all_plots(results, current_figures_dir)

        # ── Print observations ───────────────────────────────────────────────
        print_observations(results["aggregated"])

        logger.info(f"Pipeline complete for {strategy}! Check results/ for outputs.")



if __name__ == "__main__":
    main()
