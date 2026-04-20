"""
main.py — CLI Entry Point for Transformer Attention Syntactic Dependency Pipeline

Usage:
    python main.py                           # Run with default settings
    python main.py --max_depth 5             # Test up to depth 5
    python main.py --model_name bert-base-uncased --device cuda
    python main.py --num_sentences 20        # More sentences per depth
    python main.py --models bert roberta gpt2  # Multi-model comparison
    python main.py --sud_path data/sud_treebanks/SUD_English-EWT/en_ewt-sud-test.conllu

This script orchestrates the full experimental pipeline:
    1. Generate controlled recursive sentences (or load SUD treebank data)
    2. Parse gold dependency trees
    3. Extract attention weights from transformer
    4. Build attention graphs
    5. Compute alignment metrics
    6. Run statistical inference tests
    7. Generate plots and observations
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
from metrics.statistics import compute_all_statistical_tests, format_statistical_report


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
    python main.py --models bert roberta gpt2         # Multi-model comparison
    python main.py --sud_path path/to/file.conllu     # Use SUD treebank data
        """,
    )

    parser.add_argument(
        "--model_name", type=str, default=config.MODEL_NAME,
        help=f"HuggingFace model name (default: {config.MODEL_NAME})",
    )
    parser.add_argument(
        "--models", nargs='+', default=None,
        choices=list(config.AVAILABLE_MODELS.keys()),
        help="Run multi-model comparison. Options: bert, roberta, gpt2, mbert",
    )
    parser.add_argument(
        "--custom_sentences", nargs='+', default=None,
        help="Run analysis on one or more custom sentences instead of generated sentences.",
    )
    parser.add_argument(
        "--sud_path", type=str, default=None,
        help="Path to a SUD CoNLL-U treebank file for naturalistic data.",
    )
    parser.add_argument(
        "--sud_language", type=str, default=None,
        choices=list(config.SUD_TREEBANKS.keys()),
        help="Use a pre-configured SUD treebank by language name.",
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
    parser.add_argument(
        "--skip_stats", action="store_true",
        help="Skip statistical inference tests.",
    )

    return parser.parse_args()


def _resolve_sud_path(args) -> str:
    """Resolve SUD treebank path from arguments."""
    if args.sud_path:
        return args.sud_path
    if args.sud_language:
        path = config.SUD_TREEBANKS.get(args.sud_language)
        if path and os.path.exists(path):
            return path
        else:
            raise FileNotFoundError(
                f"SUD treebank not found for '{args.sud_language}' at: {path}\n"
                f"Download from https://surfacesyntacticud.github.io/ and update config.py"
            )
    return None


def main():
    """Run the full experimental pipeline."""
    args = parse_args()

    setup_logging()
    set_seed(args.seed)

    logger = logging.getLogger("main")

    # ── Determine models to run ──────────────────────────────────────────────
    if args.models:
        models_to_run = {
            name: config.AVAILABLE_MODELS[name] for name in args.models
        }
    else:
        models_to_run = {"bert": args.model_name}

    # ── Determine SUD data path ──────────────────────────────────────────────
    sud_path = _resolve_sud_path(args)

    strategies_to_run = ["mst", "top_k"] if args.pruning == "both" else [args.pruning]
    base_results_dir = args.results_dir

    for model_label, model_name in models_to_run.items():
        for strategy in strategies_to_run:
            # Separate outputs by model and pruning strategy
            if len(models_to_run) > 1:
                current_results_dir = os.path.join(base_results_dir, model_label, strategy)
            else:
                current_results_dir = os.path.join(base_results_dir, strategy)
            current_figures_dir = os.path.join(current_results_dir, "figures")

            # ── Print configuration ──────────────────────────────────────
            print("=" * 70)
            print(f"TRANSFORMER ATTENTION & SYNTACTIC DEPENDENCY ANALYSIS")
            print(f"  Model: {model_name} ({model_label.upper()}), Pruning: {strategy.upper()}")
            print("=" * 70)
            print(f"  Max depth:       {args.max_depth}")
            print(f"  Sentences/depth: {args.num_sentences}")
            print(f"  Pruning:         {strategy} (top_k={args.top_k}, "
                  f"threshold={args.threshold})")
            print(f"  Device:          {args.device}")
            print(f"  Seed:            {args.seed}")
            print(f"  SUD data:        {sud_path or 'None (using generated data)'}")
            print(f"  Results dir:     {current_results_dir}")
            print(f"  Figures dir:     {current_figures_dir}")
            print("=" * 70)

            # ── Run experiment ───────────────────────────────────────────
            logger.info(f"Starting experiment: {model_label}/{strategy}...")

            results = run_experiment(
                model_name=model_name,
                max_depth=args.max_depth,
                num_sentences=args.num_sentences,
                seed=args.seed,
                pruning_strategy=strategy,
                top_k=args.top_k,
                attention_threshold=args.threshold,
                device=args.device,
                results_dir=current_results_dir,
                custom_sentences=args.custom_sentences,
                sud_path=sud_path,
            )

            # ── Generate visualizations ──────────────────────────────────
            logger.info(f"Generating plots for {model_label}/{strategy}...")
            generate_all_plots(results, current_figures_dir)

            # ── Statistical inference ────────────────────────────────────
            if not args.skip_stats and results.get("raw_results"):
                logger.info("Running statistical inference tests...")
                stat_results = compute_all_statistical_tests(
                    results["aggregated"], results["raw_results"]
                )
                report = format_statistical_report(stat_results)
                print(report)

                # Save statistical report
                stats_path = os.path.join(current_results_dir, "statistical_tests.txt")
                with open(stats_path, "w") as f:
                    f.write(report)
                logger.info(f"Statistical report saved to {stats_path}")

            # ── Print observations ───────────────────────────────────────
            print_observations(results["aggregated"])

            logger.info(f"Pipeline complete for {model_label}/{strategy}!")



if __name__ == "__main__":
    main()
